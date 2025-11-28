import warnings
import numpy as np  # Use NumPy for CPU operations
import cupy as cp  # Use CuPy instead of NumPy
import cupy.linalg as la # Use CuPy's linalg
from cupy import log
import cupyx.scipy.special  # Import for digamma

# Use CuPy's special functions (available in recent CuPy versions installed by RAPIDS)
try:
    from cupyx.scipy.special import digamma as cp_digamma
except ImportError:
    # Fallback if somehow unavailable (less likely with RAPIDS install)
    warnings.warn("CuPy's digamma function not found. Install scipy on the CPU as a fallback.")
    import numpy as np_cpu
    from scipy.special import digamma as cpu_digamma
    def cp_digamma(x_cp):
        return cp.asarray(cpu_digamma(cp.asnumpy(x_cp)))


# Use cuML for GPU-accelerated nearest neighbors
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

# Use float32 for optimal GPU performance
DEFAULT_FLOAT = cp.float32
NUMPY_FLOAT = np.float32 # For NumPy parts if needed

# --- Helper Functions Modified for GPU ---

def add_noise(x, intens=1e-10):
    """Adds small noise to break degeneracy. Operates on CuPy arrays."""
    # Ensure input is a CuPy array of the correct type
    if not isinstance(x, cp.ndarray):
        x = cp.asarray(x, dtype=DEFAULT_FLOAT)
    elif x.dtype != DEFAULT_FLOAT:
        x = x.astype(DEFAULT_FLOAT)
    # Generate noise on the GPU
    noise = intens * cp.random.random_sample(x.shape, dtype=DEFAULT_FLOAT)
    return x + noise

import cupy as cp
import time
import warnings

# Assuming DEFAULT_FLOAT and cuNearestNeighbors are defined elsewhere
# For testing, let's define them:
DEFAULT_FLOAT = cp.float32
# Mock cuNearestNeighbors if cuML is not installed for this specific test
# In a real scenario, this would be from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

import numpy as np
# from scipy.spatial import KDTree # For Euclidean, not directly for Chebyshev
# For Chebyshev with so few points, NumPy brute is trivial

def build_and_query_neighbors_numpy_brute(points_np, query_points_np, k):
    """
    Brute-force k-NN using NumPy for very small N.
    k is 1-indexed. Uses Chebyshev distance.
    """
    n_samples_data = points_np.shape[0]
    n_samples_query = query_points_np.shape[0]

    if n_samples_data == 0:
        return np.full(n_samples_query, np.inf, dtype=np.float32)
    if n_samples_query == 0:
        return np.array([], dtype=np.float32)

    effective_k = min(k, n_samples_data -1)
    if effective_k < 1:
        # Simplified handling for numpy version, similar logic as cupy version needed for full robustness
        return np.full(n_samples_query, np.inf, dtype=np.float32)


    diff_abs = np.abs(query_points_np[:, np.newaxis, :] - points_np[np.newaxis, :, :])
    distances = np.max(diff_abs, axis=2)

    k_for_partition = min(effective_k, distances.shape[1] - 1)
    if distances.shape[1] <= k_for_partition :
         if distances.shape[1] > 0:
            kth_distances = np.partition(distances, 0, axis=1)[:, 0]
         else:
            kth_distances = np.full(n_samples_query, np.inf, dtype=np.float32)
    else:
        partitioned_distances = np.partition(distances, kth=k_for_partition, axis=1)
        kth_distances = partitioned_distances[:, k_for_partition]

    return kth_distances

def build_and_query_neighbors_cupy_brute(points, query_points, k):
    """
    Brute-force k-NN using pure CuPy for very small N.
    k is 1-indexed (e.g., k=1 for 1st NN, k=3 for 3rd NN).
    Returns distances to k-th neighbor.
    Uses Chebyshev distance.
    """
    points = cp.asarray(points, dtype=DEFAULT_FLOAT)
    query_points = cp.asarray(query_points, dtype=DEFAULT_FLOAT)
    # No contiguity check needed as CuPy ops handle it, but good practice if feeding elsewhere

    n_samples_data = points.shape[0]
    n_samples_query = query_points.shape[0]

    if n_samples_data == 0: # No points to search in
        return cp.full(n_samples_query, cp.inf, dtype=DEFAULT_FLOAT)
    if n_samples_query == 0: # No query points
        return cp.array([], dtype=DEFAULT_FLOAT)

    # k for Kozachenko-Leonenko must be < n_samples_data
    # The 'k' passed to this function is the k_KL (1-indexed)
    # We need to find the k-th closest. In a sorted array (including self), this is at index k.
    # So, we need at least k+1 elements if self is included, or k if self is excluded.
    # The entropy function usually adjusts k to be k_eff = min(orig_k, n_samples_data - 1).
    # So, the k passed here should satisfy k < n_samples_data.
    # If k >= n_samples_data, it's an issue from the caller or k needs adjustment here.
    # For safety, let's ensure k is valid.

    effective_k = min(k, n_samples_data -1) # Ensure k < n_samples_data
    if effective_k < 1: # e.g. if original k was 1 and n_samples_data was 1.
        # This means we can't even find a 1st neighbor distinct from self if points==query_points
        # Or if original k was 0.
        # For n_samples_data=1, effective_k becomes 0.
        # In such cases, it's often best to return inf or handle upstream.
        if n_samples_data == 1 and query_points.shape[0] > 0 : # Only one point in the dataset
             if cp.array_equal(points, query_points): # Querying self
                  return cp.full(n_samples_query, 0.0 if effective_k == 0 else cp.inf, dtype=DEFAULT_FLOAT) # distance to self is 0
             else: # querying other points against a single data point
                  # Calculate distance to that single point
                  diff_abs = cp.abs(query_points - points[0, :]) # Broadcasting
                  distances_to_single_point = cp.max(diff_abs, axis=1)
                  # If k=0 (effectively), maybe return this? Or inf if k > 0 expected.
                  # Assuming k >= 1 is expected for a "neighbor"
                  return cp.full(n_samples_query, cp.inf if k > 0 else distances_to_single_point, dtype=DEFAULT_FLOAT)

        return cp.full(n_samples_query, cp.inf, dtype=DEFAULT_FLOAT)



    # print(f"CuPy Brute Prep: {knn_first_part_end - knn_start_code:.6f}s")

    # Calculate pairwise Chebyshev distances
    # points shape: (N, D), query_points shape: (M, D)
    # diff shape: (M, N, D)

    diff_abs = cp.abs(query_points[:, cp.newaxis, :] - points[cp.newaxis, :, :])
    # distances shape: (M, N)
    distances = cp.max(diff_abs, axis=2)
    calc_dist_end = time.time()
    # print(f"CuPy Brute Calc Dist: {calc_dist_end - calc_dist_start:.6f}s")

    # For each query point, find the k-th smallest distance.
    # The k-th neighbor distance will be at index `effective_k` in the sorted array of distances,
    # because index 0 is self (if query_points contains points from points),
    # index 1 is 1st NN, ..., index `effective_k` is `effective_k`-th NN.

    sort_dist_start = time.time()
    # cp.partition is faster if you only need the k-th element
    # Ensure k_for_partition is a valid index for axis 1 of 'distances'
    k_for_partition = min(effective_k, distances.shape[1] - 1)

    if distances.shape[1] <= k_for_partition : # Should not happen if effective_k < n_samples_data
        # This implies n_samples_data is very small (e.g., 1) and effective_k became 0.
        # Handle cases where distance matrix might be too small for k_for_partition.
        # e.g. if n_samples_data = 1, distances is (M, 1). k_for_partition could be 0.
        if distances.shape[1] > 0:
            kth_distances = cp.partition(distances, 0, axis=1)[:, 0] # Smallest distance
        else: # distances is empty
            kth_distances = cp.full(n_samples_query, cp.inf, dtype=DEFAULT_FLOAT)

    else:
        partitioned_distances = cp.partition(distances, kth=k_for_partition, axis=1)
        kth_distances = partitioned_distances[:, k_for_partition]

    # print(f"CuPy Brute Sort Dist: {sort_dist_end - sort_dist_start:.6f}s")

    return kth_distances

def build_and_query_neighbors_gpu(points, query_points, k):
    """Builds cuML NN index and queries k+1 neighbors (returns k-th distance)."""
    knn_start_code = time.time()
    # cuML prefers C-contiguous float32 arrays
    points = cp.asarray(points, dtype=DEFAULT_FLOAT)
    query_points = cp.asarray(query_points, dtype=DEFAULT_FLOAT)
    if not points.flags['C_CONTIGUOUS']:
        points = cp.ascontiguousarray(points)
    if not query_points.flags['C_CONTIGUOUS']:
        query_points = cp.ascontiguousarray(query_points)

    n_samples = points.shape[0]
    # Adjust k if necessary (k must be less than n_samples)
    k_query = min(k + 1, n_samples)
    if k_query <= 1 and n_samples > 1: # Handle edge case where k=0 requested
        k_query = 2
    elif n_samples <=1: # Handle case with 0 or 1 sample
         return cp.full(query_points.shape[0], cp.inf, dtype=DEFAULT_FLOAT) # Or handle as error

    # Use Chebyshev distance (metric='minkowski', p=infinity equivalent)
    # algorithm='brute' is often reliable, especially for chebyshev/high-dims

    knn_start_code_end = time.time()


    knn_setup_time = time.time()

    nn_model = cuNearestNeighbors(n_neighbors=k_query, metric='chebyshev', algorithm='brute')
    nn_model.fit(points)

    knn_setup_time_end = time.time()


    knn_query = time.time()

    distances, _ = nn_model.kneighbors(query_points, n_neighbors=k_query)

    knn_query_end = time.time()


    # Return the distance to the k-th neighbor (index k, since 0 is self)
    # If k_query < k+1 (because n_samples was small), return largest available distance

    remanining_time = time.time()

    actual_k_index = min(k, k_query - 1) # k-th neighbor is at index k

    # Handle cases where k is too large for the dataset size after fitting
    if actual_k_index >= distances.shape[1]:
         # This shouldn't happen with the min(k+1, n_samples) logic, but safeguard
         # Return the furthest distance found if k is out of bounds
         # Or cp.inf might be more appropriate
         warnings.warn(f"k={k} is >= number of samples ({n_samples}) for NN query. Returning max distance found.")
         if distances.shape[1] > 0:
            return distances[:, -1]
         else: # No neighbors found at all
            return cp.full(query_points.shape[0], cp.inf, dtype=DEFAULT_FLOAT)

    result = distances[:, actual_k_index]

    remanining_time_end = time.time()

    return result


def count_neighbors_gpu(points, query_points, r_vec):
    """
    Counts neighbors within specific radii for each query point using cuML.
    Uses a workaround by querying many neighbors and filtering by distance,
    as cuML's radius_neighbors might not efficiently handle per-point radii.
    """
    points = cp.asarray(points, dtype=DEFAULT_FLOAT)
    query_points = cp.asarray(query_points, dtype=DEFAULT_FLOAT)
    r_vec = cp.asarray(r_vec, dtype=DEFAULT_FLOAT).ravel() # Ensure r is a flat CuPy array

    if not points.flags['C_CONTIGUOUS']: points = cp.ascontiguousarray(points)
    if not query_points.flags['C_CONTIGUOUS']: query_points = cp.ascontiguousarray(query_points)

    n_samples = points.shape[0]
    n_query = query_points.shape[0]

    if n_samples <= 1:
        return cp.zeros(n_query, dtype=cp.int32)

    # --- Workaround using k-neighbors ---
    # Choose a large enough k to likely capture points within radius r
    # This is a heuristic and might be inefficient/inaccurate if r is very large
    # Maximize k based on samples, but cap it to avoid excessive memory use
    k_large = min(n_samples, max(100, int(n_samples))) # Heuristic: min(100, 10% of N)
    k_query = min(k_large + 1, n_samples) # Query k_large neighbors + self

    warnings.warn("Using potentially inefficient k-NN workaround for count_neighbors_gpu.")

    nn_model = cuNearestNeighbors(n_neighbors=k_query, metric='chebyshev', algorithm='brute')
    nn_model.fit(points)
    distances, _ = nn_model.kneighbors(query_points, n_neighbors=k_query)

    # Compare distances (excluding self, column 0) to the radius for each query point
    # Need broadcasting: distances (n_query, k_query), r_vec (n_query,) -> r_vec[:, None] (n_query, 1)
    # Add small epsilon to radius for strict inequality check if needed (original used dvec - eps)
    # The original comparison was implicitly digamma(count), where count includes self.
    # Let's count points *strictly less than* r (since original subtracted eps from dvec)
    counts = cp.sum(distances[:, 1:] < r_vec[:, None], axis=1, dtype=cp.int32)

    # Add 1 to the count because digamma was applied to N_x (which includes the point itself)
    counts += 1

    return counts





def avgdigamma_gpu(points, dvec):
    """GPU version of avgdigamma using cuML NN and CuPy's digamma."""
    points = cp.asarray(points, dtype=DEFAULT_FLOAT)
    dvec = cp.asarray(dvec, dtype=DEFAULT_FLOAT)

    # Ensure dvec > 0 for radius counts
    dvec = cp.maximum(dvec, 1e-15)
    dvec_minus_eps = dvec - 1e-15 # Use slightly smaller radius for counting < r

    # Count neighbors on GPU using the workaround function
    # Pass the vector of distances dvec as the radii
    num_points = count_neighbors_gpu(points, points, dvec_minus_eps)

    # Apply digamma on the GPU, handle potential zero counts
    # Ensure num_points >= 1 for digamma (as original code implies count includes self)
    #num_points = cp.maximum(num_points, 1) # Ensure at least 1 for digamma
    return cp.mean(cp_digamma(num_points.astype(DEFAULT_FLOAT))) # Ensure float for digamma


def get_unique_probs_gpu(x):
    """GPU version using CuPy's unique."""
    x = cp.asarray(x) # Keep original type for unique

    if x.size == 0:
         return cp.array([], dtype=DEFAULT_FLOAT), cp.array([], dtype=cp.int32)

    # Workaround for multi-column unique counts using void view
    if x.ndim > 1 and x.shape[1] > 1:
        x_cont = cp.ascontiguousarray(x)
        dt = cp.dtype((cp.void, x_cont.dtype.itemsize * x_cont.shape[1]))
        x_view = x_cont.view(dt).ravel()
        _, unique_inverse, unique_counts = cp.unique(x_view, return_inverse=True, return_counts=True)
    else: # Handle 1D array or single column 2D array
        _, unique_inverse, unique_counts = cp.unique(x.ravel(), return_inverse=True, return_counts=True)

    # Calculate probabilities on GPU
    probabilities = unique_counts.astype(DEFAULT_FLOAT) / x.shape[0]
    return probabilities, unique_inverse


# --- Core Estimators Modified for GPU ---

def entropy(x, k=3, base=2, basic=False):
    """ Optimized GPU K-L k-nearest neighbor continuous entropy estimator (Input: NumPy or CuPy)"""
    entropy_start_codes = time.time()
    x = cp.asarray(x, dtype=DEFAULT_FLOAT) # Ensure x is on GPU

    # --- Handle 1D Input ---
    if x.ndim == 1:
        x = x.reshape(-1, 1) # Reshape to (n_elements, 1)
    # -----------------------

    n_elements, n_features = x.shape

    if n_elements <= 1:
        warnings.warn("n_elements <= 1. Returning NaN.")
        return cp.nan

    # Adjust k if necessary
    if k >= n_elements:
         warnings.warn(f"k={k} >= n_elements={n_elements}. Setting k={n_elements-1}. Entropy might be unreliable.")
         k = n_elements - 1 # k must be less than n_elements
    if k < 1:
        warnings.warn(f"k={k} < 1. Returning NaN.")
        return cp.nan

    # --- Add noise on GPU (ensure add_noise is GPU native) ---
    x = add_noise(x)

    entropy_start_codes_end = time.time()
    #("entropy first part",entropy_start_codes_end - entropy_start_codes)

    # --- Build index and query neighbors on GPU ---
    # Ensure build_and_query_neighbors_gpu is efficient and returns k-th dist

    knn_start = time.time()

    if basic:
      #nn_distances = build_and_query_neighbors_cupy_brute(x, x, k)
      points_np = cp.asnumpy(x)
      # query_points_np will be the same if doing self-entropy
      distances_np = build_and_query_neighbors_numpy_brute(points_np, points_np, k) # k_eff is adjusted k
      nn_distances = cp.asarray(distances_np, dtype=DEFAULT_FLOAT)
    else:
      nn_distances = build_and_query_neighbors_gpu(x, x, k)

    # Ensure distances are positive for log
    nn_distances = cp.maximum(nn_distances, cp.finfo(DEFAULT_FLOAT).eps) # Use machine epsilon
    knn_end = time.time()
    #print("knn time",knn_end - knn_start)
    # --- Calculations on GPU ---
    # Use cupyx.scipy.special.digamma and cp.log
    # Cast inputs to digamma if needed, often accepts Python int/float
    digamma_start = time.time()
    digamma_n = cupyx.scipy.special.digamma(float(n_elements))
    digamma_k = cupyx.scipy.special.digamma(float(k))

    digamma_end = time.time()
    #print("digamma time",digamma_end - digamma_start)

    # Using cp.log for all logarithms
    calculations_start = time.time()

    log_term = cp.log(nn_distances).mean()
    log_base = cp.log(cp.array(base, dtype=DEFAULT_FLOAT)) # Ensure base log is on GPU
    log_2 = cp.log(cp.array(2.0, dtype=DEFAULT_FLOAT))     # Ensure log(2) is on GPU

    # Calculate entropy in nats first
    h_nats = digamma_n - digamma_k + n_features * log_2 + n_features * log_term

    # Convert to desired base
    h = h_nats / log_base

    calculations_end = time.time()
    #print("calculations time",calculations_end - calculations_start)

    return h # Result is a CuPy scalar

def mi(x, y, z=None, k=3, base=2):
    """ GPU Mutual information (Input: NumPy or CuPy arrays) """
    x = cp.asarray(x, dtype=DEFAULT_FLOAT)
    y = cp.asarray(y, dtype=DEFAULT_FLOAT)
    if not len(x) == len(y): raise ValueError("x and y must have same length")
    n_elements = len(x)

    if k >= n_elements:
         warnings.warn(f"k={k} >= n_elements={n_elements}. Setting k={n_elements-1}. MI might be unreliable.")
         k = n_elements - 1
    if k < 1:
        warnings.warn(f"k={k} is too small. Returning NaN.")
        return cp.nan

    x = x.reshape(n_elements, -1)
    y = y.reshape(n_elements, -1)

    x = add_noise(x)
    y = add_noise(y)

    points_list = [x, y]
    if z is not None:
        z = cp.asarray(z, dtype=DEFAULT_FLOAT)
        z = z.reshape(n_elements, -1)
        if not len(z) == n_elements: raise ValueError("z must have same length as x and y")
        points_list.append(z)

    # Concatenate on GPU
    points = cp.hstack(points_list)

    # Find nearest neighbors in joint space on GPU
    dvec = build_and_query_neighbors_gpu(points, points, k)
    # Ensure dvec > 0 for avgdigamma counts
    dvec = cp.maximum(dvec, 1e-15)

    # Arguments for digamma need to be float
    k_float = cp.array(k, dtype=DEFAULT_FLOAT)
    n_elements_float = cp.array(n_elements, dtype=DEFAULT_FLOAT)

    if z is None:
        a = avgdigamma_gpu(x, dvec)
        b = avgdigamma_gpu(y, dvec)
        c = cp_digamma(k_float)
        d = cp_digamma(n_elements_float)
    else:
        xz = cp.hstack((x, z))
        yz = cp.hstack((y, z))
        a = avgdigamma_gpu(xz, dvec)
        b = avgdigamma_gpu(yz, dvec)
        c = avgdigamma_gpu(z, dvec)
        d = cp_digamma(k_float) # Original code used digamma(k) here

    # Final calculation on GPU
    mi_val = (-a - b + c + d) / log(cp.array(base, dtype=DEFAULT_FLOAT))
    # MI must be non-negative, clamp potential small negative values due to estimation noise
    return cp.maximum(0.0, mi_val) # Result is a CuPy scalar


def entropyd(sx, base=2):
    """GPU Discrete entropy estimator (Input: NumPy or CuPy)"""
    sx = cp.asarray(sx) # Ensure CuPy array, keep original type for unique
    if sx.size == 0:
        return cp.array(0.0, dtype=DEFAULT_FLOAT)

    proba, _ = get_unique_probs_gpu(sx)
    # Filter probabilities <= 0 on GPU
    proba = proba[proba > 1e-15] # Use epsilon for safety
    if proba.size == 0:
        return cp.array(0.0, dtype=DEFAULT_FLOAT)

    # Calculate entropy on GPU
    h = cp.sum(proba * cp.log(1.0 / proba)) / log(cp.array(base, dtype=DEFAULT_FLOAT))
    return h

def label_util(y, n_elements):
      # --- Iterate over unique y values (CPU loop, GPU operations inside) ---
    # Keep y_unique and y_proba on GPU, iterate using index `i`
        # --- Prepare for H(X|Y) ---
    if y.ndim == 1:
        y_reshaped = cp.expand_dims(y, axis=1) # Work with 2D shape internally
    else:
        y_reshaped = y

    _y_effective_labels_for_sorting = cp.empty(n_elements, dtype=y_reshaped.dtype)
    if y_reshaped.shape[1] == 1:
        _y_effective_labels_for_sorting = y_reshaped.ravel()
    else: # y_reshaped.shape[1] > 1
          # Original mask: (y_reshaped == yval_gpu).all(axis=1)
          # where yval_gpu is cp.array([target_scalar_class])
          # This means a row in y_reshaped matches if all its elements equal target_scalar_class.
        _non_match_label = cp.array(-1, dtype=_y_effective_labels_for_sorting.dtype) # A label that target_scalar_class won't take

        # Check if all elements in a row are the same as the first element of that row
        _row_elements_all_same_as_first = cp.all(y_reshaped == y_reshaped[:, :1], axis=1)

        # The effective label is the first element if all row elements are same, otherwise it's _non_match_label.
        _y_effective_labels_for_sorting = cp.where(
            _row_elements_all_same_as_first,
            y_reshaped[:, 0], # The common value in the row
            _non_match_label  # This row won't match any yval_gpu via the .all(axis=1) logic if elements differ
        )
    _sort_indices = cp.argsort(_y_effective_labels_for_sorting)

    _sorted_effective_labels = _y_effective_labels_for_sorting[_sort_indices]

    # Create a lookup map: {effective_label_value: (start_index_in_sorted_arrays, count)}
    # This map only stores info for labels actually present and relevant.
    _class_data_lookup_map = {}

    if n_elements > 0: # Proceed only if there's data
      _actual_unique_eff_labels, _L_start_indices, _L_counts = cp.unique(
        _sorted_effective_labels, return_index=True, return_counts=True
      )
      for i_actual in range(len(_actual_unique_eff_labels)):
          label_val = _actual_unique_eff_labels[i_actual].item()
          # We only care about labels that could match yval_gpu[0] (which are 0 to number_of_class-1).
          # The _non_match_label (e.g., -1) will be ignored here if not in that range.
          if label_val >= 0: # Assuming class labels from y_unique are non-negative
              start_idx = _L_start_indices[i_actual].item()
              count = _L_counts[i_actual].item()
              _class_data_lookup_map[label_val] = (start_idx, count)

    return _class_data_lookup_map, _sort_indices
def micd(x, y, k=3, base=2, warning_flag=True, number_of_class=10, _class_data_lookup_map=None, _sort_indices=None):
    """ Optimized GPU Mixed Mutual Information (x continuous, y discrete) (Input: NumPy or CuPy) """

    x = cp.asarray(x, dtype=DEFAULT_FLOAT)
    y = cp.asarray(y) # Keep original type for unique operations
    if not len(x) == len(y): raise ValueError("x and y must have same length")
    n_elements = len(x)
    if n_elements == 0: return cp.array(0.0, dtype=DEFAULT_FLOAT)

    # --- Calculate H(X) on GPU ---
    k_hx = min(k, n_elements - 1) if n_elements > 1 else 0
    if k_hx < 1:
        entropy_x = cp.array(0.0, dtype=DEFAULT_FLOAT)
    else:
        # Ensure entropy function is efficient on GPU
        entropy_x = entropy(x, k_hx, base)

    # --- Prepare for H(X|Y) ---
    if y.ndim == 1:
        y_reshaped = cp.expand_dims(y, axis=1) # Work with 2D shape internally
    else:
        y_reshaped = y

    # --- Get unique values, indices, and counts for y on GPU ---
    # Use axis=0 for multi-column unique if applicable
    '''
        if y_reshaped.shape[1] == 1: # Single column
        y_unique_flat, y_inverse, y_counts = cp.unique(y_reshaped.ravel(), return_inverse=True, return_counts=True)
        # Reshape unique back to column vector (n_unique, 1)
        y_unique = cp.expand_dims(y_unique_flat, axis=1)
    else: # Multi-column
        # Use axis=0 for potentially better performance/readability
        y_unique, y_inverse, y_counts = cp.unique(y_reshaped, axis=0, return_inverse=True, return_counts=True)
    '''


    #y_unique is 2d array like this [[0], [1], [2], ].. please create it with number_of_class
    y_unique= np.arange(number_of_class).reshape(-1, 1)

    y_unique = cp.asarray(y_unique)

    #y_proba = y_counts.astype(DEFAULT_FLOAT) / n_elements
    #y_proba is 1/number_of_class, for each class
    y_proba = cp.ones(number_of_class) / number_of_class
    n_unique = len(y_unique)

    # --- Calculate conditional entropy H(X|Y) ---
    entropy_x_given_y = cp.array(0.0, dtype=DEFAULT_FLOAT)


    total_time = 0
    prepate_time = 0

    _x_sorted_for_lookup = x[_sort_indices] # x_gpu is the CuPy array from input x


    ALL_LOOP_TIME = time.time()
    for i in range(n_unique):
        # Access unique value and probability directly from GPU arrays
        p_time_start = time.time()
        yval_gpu = y_unique[i] # This is a CuPy scalar or 1D CuPy array (row)
        py = y_proba[i]        # This is a CuPy scalar

        # Create boolean mask on GPU using direct GPU comparisons
        # Compare y_reshaped (original y data) with the current unique row yval_gpu
        current_target_class_scalar = yval_gpu[0].item() # This is the integer class value, e.g., i

        if current_target_class_scalar in _class_data_lookup_map:
            # This class was found among the effective labels derived from y_reshaped
            start_idx, count = _class_data_lookup_map[current_target_class_scalar]
            x_given_y = _x_sorted_for_lookup[start_idx : start_idx + count]
        else:
            # This class (from 0 to number_of_class-1) is not present as an
            # effective label in the input y_reshaped data.
            x_given_y = cp.array([], dtype=x.dtype)

        '''
        if y_reshaped.shape[1] == 1:
            # yval_gpu is shape (1,) -> access scalar element for comparison
            mask = (y_reshaped.ravel() == yval_gpu[0])
        else:
            # yval_gpu is shape (n_features,) -> broadcast comparison
            mask = (y_reshaped == yval_gpu).all(axis=1) # Compare row-wise

        # Filter x on GPU using the mask
        ##BOTTLENECK
        x_given_y = x[mask]
        ##BOTTLENECK
        '''

        n_subset = x_given_y.shape[0] # More robust way to get length



        # --- Calculate H(X|Y=y_i) ---
        if n_subset <= 1:
            # Handle insufficient data case - Using H(X) as upper bound
            if warning_flag:
                 # Convert yval_gpu to numpy for printing *only if* warning needed
                 yval_np = cp.asnumpy(yval_gpu)
                 warnings.warn(f"Warning: After conditioning on y={yval_np}, insufficient data ({n_subset}). Assuming maximal conditional entropy H(X) for this subset.")
            entropy_x_given_y += py * entropy_x # Add contribution weighted by p(y)
            continue

        # Determine k for the subset, ensuring k >= 1
        k_subset = min(k, n_subset - 1)

        p_time_end = time.time()
        prepate_time += (p_time_end - p_time_start)

        if k_subset < 1:
            if warning_flag:
                 yval_np = cp.asnumpy(yval_gpu)
                 warnings.warn(f"Warning: After conditioning on y={yval_np}, k={k} is too large for subset size {n_subset} (k_subset<1). Assuming maximal conditional entropy H(X) for this subset.")
            entropy_x_given_y += py * entropy_x
        else:
            # Calculate entropy for the subset on GPU
            time_start = time.time()
            h_xgiveny_val = entropy(x_given_y, k_subset, base, basic = False)

            # Accumulate on GPU
            entropy_x_given_y += py * h_xgiveny_val # Accumulate on GPU
            time_end = time.time()
            total_time += (time_end - time_start)
    #print('total time',total_time)
    #print('prepare_time', prepate_time)
    ALL_LOOP_TIME_END = time.time()
    #print('ALL_LOOP_TIME', ALL_LOOP_TIME_END - ALL_LOOP_TIME)

    # --- Final Mutual Information ---
    # Ensure non-negativity

    mi_cd = cp.maximum(cp.array(0.0, dtype=DEFAULT_FLOAT), entropy_x - entropy_x_given_y)
    return mi_cd


import cupy as cp
import cupy.linalg as la
from cupy import log
import cupyx.scipy.special
import numpy as np
import warnings
import math

# ==============================================================================
# Assume all necessary helper/core functions (like mi, micd, etc.) are defined
# correctly as in your provided code above this point.
# ==============================================================================

def generate_random_subspace_basis(d, k_slice_dim, dtype=DEFAULT_FLOAT):
    """
    Generates an orthonormal basis for a random k-subspace in d dimensions.
    Uses QR decomposition of Gaussian vectors.
    """
    if not isinstance(k_slice_dim, int) or k_slice_dim <= 0:
         raise ValueError("k_slice_dim must be a positive integer.")
    if k_slice_dim > d:
        raise ValueError(f"Slice dimension k_slice_dim ({k_slice_dim}) cannot be larger than original dimension d ({d}).")
    if k_slice_dim == d:
        return cp.eye(d, dtype=dtype)

    # Generate k_slice_dim random vectors in d dimensions
    random_vectors = cp.random.randn(d, k_slice_dim, dtype=dtype)
    # Orthonormalize using QR decomposition
    basis, _ = la.qr(random_vectors, mode='reduced') # Shape (d, k_slice_dim)
    return basis


def compute_ksmi_optimized_parallel(x, y, k_slice_dim=5, m=500, method='ksg', number_of_class=80,
                                  discrete=True, k_nn=3, base=2,
                                   num_streams=100): 
    """
    Optimized k-SMI estimator using CUDA streams for parallel MI calculation.
    Precomputes all projections, then launches MI computations concurrently.

    Args:
        x (cp.ndarray): Continuous input data (n_samples, n_features_x).
        y (cp.ndarray): Second input data (discrete or continuous).
        k_slice_dim (int): The slicing dimension 'k' for k-SMI (1 <= k <= n_features_x).
        m (int): Number of random k-dimensional projections to average over.
        method (str): MI estimation method. MUST be 'ksg' for this optimized version.
        discrete (bool): Whether y should be treated as discrete (uses micd) or continuous (uses mi).
        k_nn (int): Number of nearest neighbors for the internal KSG MI estimators.
        base (int): Base for the logarithm in MI calculation.
        num_streams (int): Number of CUDA streams to use for parallel execution.
                           Adjust based on GPU memory and workload.

    Returns:
        float: Estimated k-Sliced Mutual Information (k-SMI).

    Raises:
        ValueError: If method is not 'ksg', or inputs are invalid.
        cp.cuda.MemoryError: If intermediate matrices or concurrent tasks exceed GPU memory.
    """
    print(f"Computing k-SMI with {num_streams} streams...")
    if method != 'ksg':
        raise ValueError("This optimized function currently only supports method='ksg'.")

    # --- Input Validation and Preparation ---
    if not isinstance(x, cp.ndarray): x = cp.asarray(x, dtype=DEFAULT_FLOAT)
    elif x.dtype != DEFAULT_FLOAT: x = x.astype(DEFAULT_FLOAT)
    if x.ndim == 1: x = x.reshape(-1, 1)
    n_elements, n_features_x = x.shape

    if not (1 <= k_slice_dim <= n_features_x):
         raise ValueError(f"k_slice_dim ({k_slice_dim}) must be between 1 and n_features_x ({n_features_x}).")

    # Prepare y
    y = cp.asarray(y)
    if not discrete:
        if y.dtype != DEFAULT_FLOAT: y = y.astype(DEFAULT_FLOAT)
        if y.ndim == 1: y = y.reshape(-1, 1)
    else:
        # Ensure y is contiguous for potential internal operations if discrete
        if y.ndim == 1: y = y.reshape(-1, 1)
        if not y.flags['C_CONTIGUOUS']: y = cp.ascontiguousarray(y)

    if not len(x) == len(y): raise ValueError("x and y must have same length")

    min_samples_needed = k_nn + 1
    if n_elements < min_samples_needed:
        warnings.warn(f"Number of samples ({n_elements}) is less than k_nn+1 ({min_samples_needed}). "
                      f"Cannot compute KSG MI reliably. Returning 0.0")
        return 0.0

    # --- OPTIMIZATION: Precompute all bases and projections ---
    try:
        print("Precomputing projections...")
        # 1. Generate all bases and stack horizontally
        all_bases_list = [generate_random_subspace_basis(n_features_x, k_slice_dim, dtype=DEFAULT_FLOAT) for _ in range(m)]
        all_bases = cp.hstack(all_bases_list) # Shape: (n_features_x, k_slice_dim * m)
        del all_bases_list

        # 2. Perform one large batch projection
        # x: (n, d), all_bases: (d, k*m) -> x_proj_batch: (n, k*m)
        x_proj_batch = cp.dot(x, all_bases)

        del all_bases
        cp.get_default_memory_pool().free_all_blocks()
        print("Projections computed.")

    except cp.cuda.MemoryError as e:
        print(f"GPU Memory Error during precomputation: {e}")
        print("Try reducing 'm' (number of projections) or 'k_slice_dim'.")
        raise
    except Exception as e:
        print(f"An error occurred during precomputation: {e}")
        raise

    # --- Parallel MI Calculation using CUDA Streams ---
    print(f"Launching {m} MI calculations across {num_streams} streams...")
    mi_array = [] # Store the future results (CuPy arrays)

    # Make y read-only accessible to all streams
    y.flags.writeable = False # Optional optimization if y is large

    if discrete:
       _class_data_lookup_map, _sort_indices = label_util(y,len(x))

    for i in range(m):
      try:
          # Slice the data for this projection *within the stream context*
          start_col = i * k_slice_dim
          end_col = (i + 1) * k_slice_dim
          # Ensure slice is contiguous for MI functions
          # Note: Slicing might incur some overhead, but unavoidable here.
          x_proj_i = cp.ascontiguousarray(x_proj_batch[:, start_col:end_col])

          # --- Launch the appropriate MI calculation asynchronously ---
          if discrete:
              # Note: warning_flag=False is important to avoid flooding console
              #convert to numpy array
              #x_proj_i = cp.asnumpy(x_proj_i)
              #y = cp.asnumpy(y)
              #mi_result = micd_batch_parallel(x_proj_i, y, k=k_nn, base=base, warning_flag=False)
              micd_start = time.time()
              mi_result = micd(x_proj_i, y, k=k_nn, base=base, warning_flag=True, number_of_class=number_of_class,_class_data_lookup_map=_class_data_lookup_map, _sort_indices=_sort_indices)
              micd_end = time.time()
              #print(f"MICD took: {micd_end - micd_start:.4f} seconds")
          else:
              mi_result = mi(x_proj_i, y, k=k_nn, base=base)
      

          # Append the CuPy scalar (representing the future result)
          mi_array.append(mi_result)

      except Exception as e:
          warnings.warn(f"Error calculating MI for projection {i}: {e}. Skipping with NaN.")
          # Append NaN, ensure it's a CuPy scalar on the default stream context maybe?
          # It's better to append directly within the stream context if possible,
          # but for error handling, creating it outside might be safer.
          # Let's ensure it's a CuPy NaN.
          mi_array.append(cp.array(cp.nan, dtype=DEFAULT_FLOAT))

    end_time_to_launch = time.time()

    #convert mi_array to cp array
    mi_array = cp.array(mi_array)



    # --- Synchronization and Aggregation ---
    print("Synchronizing streams and collecting results...")
    # Method 1: Synchronize all streams first (often recommended)


    # Now all computations are done, results in mi_futures are ready on GPU.
    # Check if any futures were actually generated
    # Calculate mean, ignoring NaNs
    valid_mi_mask = ~cp.isnan(mi_array)
    num_valid = cp.sum(valid_mi_mask).item() # Use item() to get Python int
    num_total = len(mi_array)

    if num_valid > 0:
        ksmi_val = cp.mean(mi_array[valid_mi_mask])
        print(f"Calculation complete. Averaged {num_valid}/{num_total} valid MI results.")
        # Return standard Python float
        return ksmi_val.item()
    else:
        warnings.warn(f"All {num_total} MI calculations resulted in NaN or errors.")
        # Return numpy NaN for final Python float output consistent with errors
        return np.nan