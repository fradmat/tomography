import numpy as np
import sys
import time

def grid_60_40_bounded(bdmatrix):
    # R = np.linspace(-.5, .5, 40)
    # z = np.linspace(-.75, .75, 60)
    R = np.linspace(-.55, .45, 40)
    z = np.linspace(-.85, .65, 60)
    #Rv, zv = np.meshgrid(R, z)
    # bdmat = np.logical_not(np.load('Emissivity_0.000-0.100_rec_30382.npz')['BdMat'])
    # print(bdmat.shape)
    pixels = []
    for j in range(60):
        for k in range(40):
            if bdmatrix[j,k]:
                pixels += [Pixel(R[k], z[j])]
    return pixels

def grid_60_40_bounded_poloidal(bdmatrix):
    # R = np.linspace(-.5, .5, 40)
    # z = np.linspace(-.75, .75, 60)
    R = np.linspace(-.55, .45, 40)
    z = np.linspace(-.85, .65, 60)
    R = np.linspace(-.5, .5, 40)
    z = np.linspace(-.75, .75, 60)
    #Rv, zv = np.meshgrid(R, z)
    # bdmat = np.logical_not(np.load('Emissivity_0.000-0.100_rec_30382.npz')['BdMat'])
    # print(bdmat.shape)
    # print(R)
    # print(z)
    pixels = []
    for j in range(60):
        for k in range(40):
            if bdmatrix[j,k]:
                r, Z = R[k], z[j]
                pixels += [RadialPixel(np.sqrt(r**2 + Z**2), np.arctan2(Z, r)+np.pi)]
                # pixels += [np.sqrt(pixel.R**2 + pixel.z**2), (np.arctan2(pixel.z, pixel.R)+np.pi)]
    return pixels
    
class Pixel(object):
    def __init__(self, R, z):
        self.R = R
        self.z = z
        
class RadialPixel(object):
    def __init__(self, r, phi):
        self.r = r
        self.phi = phi
        
class GaussianProcess():
    def __init__(self,geo_matrix, prior_mean,grid,kernel):
        self.geo_matrix = geo_matrix#[:, self.bdmat]
        print('Geometric Matrix:', self.geo_matrix.shape)
        self.geo_matrix_transpose = np.transpose(self.geo_matrix)
        print('Transpose Geometric Matrix:', self.geo_matrix_transpose.shape)
        self.prior_mean = prior_mean
        print('Prior Mean:', self.prior_mean, self.prior_mean.shape)
        self.grid = grid
        print('Computing solutions on grid of size:', len(grid))
        self.kernel = kernel
        self.kernels = {'exponential': self.kernel_exponential}
        
    def kernel_exponential(self,):
        raise NotImplementedError("Please Implement this method")
    
    def marginal_log_likelihood(self, y, fL, cov, inv_cov):
        return -.5*(208*np.log(2*np.pi)+np.linalg.slogdet(cov)[1]+np.transpose(y-fL).dot(inv_cov).dot(y-fL))
    
    def calc_distances():
        raise NotImplementedError("Please Implement this method")
    
    def compute_MAP_mats(self, sigma_err, sigma_f, sigma_x, inf_err_mask, measurement):
        rec_start = time.time()
        inf_err_mask = inf_err_mask.astype(bool)
        filtered_measurement = measurement[inf_err_mask]
        filt_geo_m = self.geo_matrix[inf_err_mask]
        filt_geo_m_t = self.geo_matrix_transpose[:,inf_err_mask]
        fL = filt_geo_m.dot(self.prior_mean)
        pr_cov = self.pr_covs[sigma_f,sigma_x,:,:]
        sigma_y = np.zeros((len(filtered_measurement), len(filtered_measurement)))
        sigma_err_val = self.sigma_errs[sigma_err]
        err_vals = np.maximum((1./9)*(filtered_measurement*sigma_err_val)**2, np.ones(len(filtered_measurement)))
        np.fill_diagonal(sigma_y, err_vals)
        
        K_ll = filt_geo_m.dot(pr_cov).dot(filt_geo_m_t)
        sigma_d = (K_ll + sigma_y).astype(np.float64)
        try:
            sigma_d_inv = np.linalg.solve(sigma_d, np.identity(len(filtered_measurement))).astype(np.float64)
        except:
            sigma_d_inv = np.linalg.lstsq(sigma_d, np.identity(len(filtered_measurement)))[0]
        rec_end = time.time()
        best_post_mean, best_post_cov, t_delta = self.compute_MAP(pr_cov, inf_err_mask, sigma_d_inv, filtered_measurement, fL, filt_geo_m, filt_geo_m_t)
        
        rec_delta = rec_end-rec_start + t_delta
        return best_post_mean, best_post_cov, sigma_d, rec_delta

        
    def compute_MAP(self, pr_cov, inf_err_mask, sigma_d_inv, filtered_measurement, fL, filt_geo_m, filt_geo_m_t):
        # print(self.prior_mean.shape, filtered_measurement.shape,  fL.shape, sigma_d_inv.shape, self.geo_matrix_transpose[:,inf_err_mask].shape)
        start = time.time()
        m_t = pr_cov.dot(filt_geo_m_t).dot(sigma_d_inv)
        best_post_mean = self.prior_mean + m_t.dot(filtered_measurement - fL)
        best_post_cov = pr_cov - m_t.dot(filt_geo_m).dot(pr_cov)
        end = time.time()
        return best_post_mean, best_post_cov, end-start

class GaussianProcessEuclidean(GaussianProcess):
    def __init__(self, geo_matrix, prior_mean, grid, #boundary_matrix,
                 kernel='exponential',
                 sigma_fs = np.logspace(6,1,6),
                 sigma_xs = np.linspace(.045, .445, 20, endpoint=False),
                 sigma_errs = np.logspace(2, 0, 3, endpoint=True, base=2),
                 ):
        
        super().__init__(geo_matrix, prior_mean, grid, kernel)
        
        self.sigma_fs = sigma_fs
        self.sigma_xs = sigma_xs
        print('will loop over:')
        print('sigmafs', self.sigma_fs)
        print('sigmaxs', self.sigma_xs)
        self.sigma_errs = sigma_errs
        print('sigma_errs', self.sigma_errs)
        
        self.distances = self.calc_distances(self.grid)
        pr_covs = []
        for sigma_f in self.sigma_fs:
            cov_sigmaf = []
            for sigma_x in self.sigma_xs:
                cov = self.kernels[self.kernel]([sigma_f, sigma_x], self.distances)
                cov_sigmaf += [cov]
                sys.stdout.flush()
            pr_covs += [cov_sigmaf]
        self.pr_covs = np.asarray(pr_covs)
        
        
    def calc_distances(self, grid):
        distance = []
        k = 0
        for p1 in grid:
            for p2 in grid:
                distance += [np.sqrt((p1.R-p2.R)**2+(p1.z - p2.z)**2)]
            k += 1
        return np.asarray(distance).reshape(len(grid),len(grid))

    def kernel_exponential(self, thetas, distances):
        # print('exponential kernel, thetas: ', thetas)
        exp = thetas[0]**2 * np.exp(-1.*(distances**2)/(2*thetas[1]**2))
        return exp
    
    def marginalize_and_calculate_hps(self, sample_id, measurement, measurement_errors, inf_err_mask):
        error = measurement_errors
        measurement = measurement
        filtered_measurement = measurement[inf_err_mask]
        fL = self.geo_matrix[inf_err_mask].dot(self.prior_mean)
        bad_mll = -1e8
        best_post_mean = []
        best_post_cov = []
        best_digma_d = []
        start = time.time()
        best_sigma_f = np.zeros(len(self.sigma_fs))
        best_sigma_x = np.zeros(len(self.sigma_xs))
        best_sigma_err = np.zeros(len(self.sigma_errs))
        best_sigma_grid = np.zeros(len(self.sigma_fs)*len(self.sigma_xs)).reshape(len(self.sigma_fs), len(self.sigma_xs))
        # print('here', best_sigma_grid)
        # single_multiclass = np.zeros(len(self.sigma_errs) * len(self.sigma_errs) * len(self.sigma_xs))
        start = time.time()
        for h, sigma_err in enumerate(self.sigma_errs):
            sigma_y = np.zeros((len(filtered_measurement), len(filtered_measurement)))
            # get sigma_y only in positions where error is not infnite. error for negative measurements should have been increased by here. 
            # np.fill_diagonal(sigma_y, (1./9)*(np.abs(error[inf_err_mask])*sigma_err)**2)
            # err_vals = np.maximum((1./9)*(np.abs(error[inf_err_mask])*sigma_err)**2, np.ones(len(filtered_measurement)))
            err_vals = np.maximum((1./9)*(filtered_measurement*sigma_err)**2, np.ones(len(filtered_measurement)))
            np.fill_diagonal(sigma_y, err_vals)
            for i, sigma_f in enumerate(self.sigma_fs):
                for j, sigma_x in enumerate(self.sigma_xs):
                    pr_cov = self.pr_covs[i,j,:,:]
                    assert pr_cov.shape == (self.prior_mean.shape[0], self.prior_mean.shape[0])
    
                    K_ll = self.geo_matrix[inf_err_mask].dot(pr_cov).dot(self.geo_matrix_transpose[:,inf_err_mask])
                    sigma_d = (K_ll + sigma_y).astype(np.float64)
                    try:
                        sigma_d_inv = np.linalg.solve(sigma_d, np.identity(len(filtered_measurement))).astype(np.float64)
                    except:
                        sigma_d_inv = np.linalg.lstsq(sigma_d, np.identity(len(filtered_measurement)))[0]
                    mll = self.marginal_log_likelihood(y=filtered_measurement, fL=fL, cov=sigma_d, inv_cov=sigma_d_inv)
                    # print("%10.10f  %10.10f    %10.10f  %10.10f " % (sigma_f, sigma_x, sigma_err,  mll))
                    if mll > bad_mll:
                        # mask = np.zeros(2400)
                        bad_mll = mll
                        best_hps = {u'. \u03C3err':sigma_err, u'. \u03C3f':sigma_f, u'. \u03C3x':sigma_x}
                        # best_post_mean = self.prior_mean + pr_cov.dot(self.geo_matrix_transpose[:,inf_err_mask]).dot(sigma_d_inv).dot(filtered_measurement - fL)
                        # best_post_cov = pr_cov - pr_cov.dot(self.geo_matrix_transpose[:,inf_err_mask]).dot(sigma_d_inv).dot(self.geo_matrix[inf_err_mask]).dot(pr_cov)
                        best_post_mean, best_post_cov, t_delta = self.compute_MAP(pr_cov, inf_err_mask, sigma_d_inv, filtered_measurement, fL,
                                                                                  self.geo_matrix[inf_err_mask], self.geo_matrix_transpose[:,inf_err_mask])
                        best_sigma_d = sigma_d
                        # 
                        best_sigma_f = np.zeros(len(self.sigma_fs))
                        best_sigma_x = np.zeros(len(self.sigma_xs))
                        best_sigma_err = np.zeros(len(self.sigma_errs))
                        best_sigma_grid = np.zeros(len(self.sigma_fs)*len(self.sigma_xs)*len(self.sigma_errs)).reshape(len(self.sigma_errs), len(self.sigma_fs), len(self.sigma_xs))
                        best_sigma_f[i] = 1
                        best_sigma_x[j] = 1
                        best_sigma_err[h] = 1
                        best_sigma_grid[h, i, j] = 1
                        single_multiclass = best_sigma_grid.flatten()
                        # print(single_multiclass, np.argwhere(single_multiclass==1)[0])
                        # exit(0)
                        # print("%10.10f     %10.10f   %10.10f  %10.10f %10.10f " % (sigma_f, sigma_x, sigma_err, int(single_multiclass[0]), mll))
        end = time.time()
        print(end - start)
        t_delta = end - start
        return best_post_mean, best_post_cov, best_hps, best_sigma_d, bad_mll, best_sigma_err, best_sigma_f, best_sigma_x, best_sigma_grid, single_multiclass,t_delta   # best_sigma_err_ind, best_sigma_f_ind, best_sigma_x_ind
    
    




class GaussianProcessEuclideanErr(GaussianProcess):
    def __init__(self, geo_matrix, prior_mean, grid, #boundary_matrix,
                 kernel='exponential',
                 sigma_fs = np.logspace(6,1,6),
                 sigma_xs = np.linspace(.045, .445, 20, endpoint=False),
                 sigma_errs = np.logspace(2, 0, 3, endpoint=True, base=2),
                 sigma_d_errs = np.asarray([100, 1000])
                 ):
        
        super().__init__(geo_matrix, prior_mean, grid, kernel)
        
        self.sigma_fs = sigma_fs
        self.sigma_xs = sigma_xs
        print('will loop over:')
        print('sigmafs', self.sigma_fs)
        print('sigmaxs', self.sigma_xs)
        self.sigma_errs = sigma_errs
        print('sigma_errs', self.sigma_errs)
        self.sigma_d_errs = sigma_d_errs
        print('sigma_d_errs', self.sigma_d_errs)
        
        self.distances = self.calc_distances(self.grid)
        pr_covs = []
        for sigma_f in self.sigma_fs:
            cov_sigmaf = []
            for sigma_x in self.sigma_xs:
                cov = self.kernels[self.kernel]([sigma_f, sigma_x], self.distances)
                cov_sigmaf += [cov]
                sys.stdout.flush()
            pr_covs += [cov_sigmaf]
        self.pr_covs = np.asarray(pr_covs)
        
        
    def calc_distances(self, grid):
        distance = []
        k = 0
        for p1 in grid:
            for p2 in grid:
                distance += [np.sqrt((p1.R-p2.R)**2+(p1.z - p2.z)**2)]
            k += 1
        return np.asarray(distance).reshape(len(grid),len(grid))

    def kernel_exponential(self, thetas, distances):
        # print('exponential kernel, thetas: ', thetas)
        exp = thetas[0]**2 * np.exp(-1.*(distances**2)/(2*thetas[1]**2))
        return exp
    
    def marginalize_and_calculate_hps(self, sample_id, measurement, measurement_errors, inf_err_mask):
        error = measurement_errors
        measurement = measurement
        filtered_measurement = measurement[inf_err_mask]
        fL = self.geo_matrix[inf_err_mask].dot(self.prior_mean)
        bad_mll = -1e8
        best_post_mean = []
        best_post_cov = []
        best_digma_d = []
        start = time.time()
        # best_sigma_f = np.zeros(len(self.sigma_fs))
        # best_sigma_x = np.zeros(len(self.sigma_xs))
        # best_sigma_err = np.zeros(len(self.sigma_errs))
        # best_sigma_d_err = np.zeros(len(self.sigma_d_errs))
        best_sigma_grid = np.zeros(len(self.sigma_fs)*len(self.sigma_xs)).reshape(len(self.sigma_fs), len(self.sigma_xs))
        # print('here', best_sigma_grid)
        # single_multiclass = np.zeros(len(self.sigma_errs) * len(self.sigma_errs) * len(self.sigma_xs))
        start = time.time()
        for g, sigma_d_err in enumerate(self.sigma_d_errs):
            for h, sigma_err in enumerate(self.sigma_errs):
                sigma_y = np.zeros((len(filtered_measurement), len(filtered_measurement)))
                # get sigma_y only in positions where error is not infnite. error for negative measurements should have been increased by here. 
                # np.fill_diagonal(sigma_y, (1./9)*(np.abs(error[inf_err_mask])*sigma_err)**2)
                # err_vals = np.maximum((1./9)*(np.abs(error[inf_err_mask])*sigma_err)**2, np.ones(len(filtered_measurement)))
                # err_vals = np.maximum((1./9)*(filtered_measurement*sigma_err)**2, np.ones(len(filtered_measurement)))
                err_vals = (1./9)*(sigma_d_err+filtered_measurement*sigma_err)
                np.fill_diagonal(sigma_y, err_vals)
                for i, sigma_f in enumerate(self.sigma_fs):
                    for j, sigma_x in enumerate(self.sigma_xs):
                        pr_cov = self.pr_covs[i,j,:,:]
                        assert pr_cov.shape == (self.prior_mean.shape[0], self.prior_mean.shape[0])
        
                        K_ll = self.geo_matrix[inf_err_mask].dot(pr_cov).dot(self.geo_matrix_transpose[:,inf_err_mask])
                        sigma_d = (K_ll + sigma_y).astype(np.float64)
                        try:
                            sigma_d_inv = np.linalg.solve(sigma_d, np.identity(len(filtered_measurement))).astype(np.float64)
                        except:
                            sigma_d_inv = np.linalg.lstsq(sigma_d, np.identity(len(filtered_measurement)))[0]
                        mll = self.marginal_log_likelihood(y=filtered_measurement, fL=fL, cov=sigma_d, inv_cov=sigma_d_inv)
                        # print("%10.10f  %10.10f    %10.10f  %10.10f " % (sigma_f, sigma_x, sigma_err,  mll))
                        if mll > bad_mll:
                            # mask = np.zeros(2400)
                            bad_mll = mll
                            best_hps = {u'. \u03C3err_f':sigma_err, u'. \u03C3err_d':sigma_d_err, u'. \u03C3f':sigma_f, u'. \u03C3x':sigma_x}
                            # best_post_mean = self.prior_mean + pr_cov.dot(self.geo_matrix_transpose[:,inf_err_mask]).dot(sigma_d_inv).dot(filtered_measurement - fL)
                            # best_post_cov = pr_cov - pr_cov.dot(self.geo_matrix_transpose[:,inf_err_mask]).dot(sigma_d_inv).dot(self.geo_matrix[inf_err_mask]).dot(pr_cov)
                            best_post_mean, best_post_cov, t_delta = self.compute_MAP(pr_cov, inf_err_mask, sigma_d_inv, filtered_measurement, fL,
                                                                                      self.geo_matrix[inf_err_mask], self.geo_matrix_transpose[:,inf_err_mask])
                            best_sigma_d = sigma_d
                            best_sigma_d_err = np.zeros(len(self.sigma_d_errs))
                            best_sigma_f = np.zeros(len(self.sigma_fs))
                            best_sigma_x = np.zeros(len(self.sigma_xs))
                            best_sigma_err = np.zeros(len(self.sigma_errs))
                            best_sigma_grid = np.zeros(
                                len(self.sigma_fs)*len(self.sigma_xs)*len(self.sigma_errs)*len(self.sigma_d_errs)).reshape(
                                    len(self.sigma_d_errs), len(self.sigma_errs), len(self.sigma_fs), len(self.sigma_xs))
                            best_sigma_f[i] = 1
                            best_sigma_x[j] = 1
                            best_sigma_err[h] = 1
                            best_sigma_d_err[g] = 1
                            best_sigma_grid[g, h, i, j] = 1
                            single_multiclass = best_sigma_grid.flatten()
                            # print(single_multiclass, np.argwhere(single_multiclass==1)[0])
                            # exit(0)
                            print("%10.10f  %10.10f   %10.10f   %10.10f  %10.10f " % (sigma_f, sigma_x, sigma_err, sigma_d_err, mll))
        end = time.time()
        print(end - start)
        t_delta = end - start
        return best_post_mean, best_post_cov, best_hps, best_sigma_d, bad_mll, best_sigma_err, best_sigma_f, best_sigma_x, best_sigma_d_err, best_sigma_grid, single_multiclass,t_delta   # best_sigma_err_ind, best_sigma_f_ind, best_sigma_x_ind
    
    
  
class GaussianProcessVectorial(GaussianProcess):
    def __init__(self, geo_matrix, prior_mean, grid, #boundary_matrix,
                 kernel='exponential',
                 sigma_fs = np.logspace(6,1,6),
                 sigma_l1s = np.linspace(.045, .445, 10, endpoint=False),
                 sigma_l2s = np.linspace(.045, .445, 10, endpoint=False),
                 sigma_errs = np.logspace(2, 0, 3, endpoint=True, base=2),
                 ):
        super().__init__(geo_matrix, prior_mean, grid, kernel)
        
        self.sigma_fs = sigma_fs
        self.sigma_l1s = sigma_l1s
        self.sigma_l2s = sigma_l2s
        print('will loop over:')
        print('sigmafs', self.sigma_fs)
        print('sigmal1s', self.sigma_l1s)
        print('sigmal2s', self.sigma_l2s)
        self.sigma_errs = sigma_errs
        print('sigma_errs', self.sigma_errs)
      
        
        self.distances = self.calc_distances(self.grid)
        pr_covs = []
        
        for sigma_f in self.sigma_fs:
            cov_sigmaf = []
            for sigma_l1 in self.sigma_l1s:
                cov_sigma_l1 = []
                for sigma_l2 in self.sigma_l2s:
                    cov = self.kernels[self.kernel]([sigma_f, sigma_l1, sigma_l2], self.distances)
                    cov_sigma_l1 += [cov]
                cov_sigmaf += [cov_sigma_l1]
            pr_covs += [cov_sigmaf]    
            
        self.pr_covs = np.asarray(pr_covs)
        print('self.pr_covs', self.pr_covs.shape)
        
    def calc_distances(self, grid):
        print('computing vectorial distances')
        distances = []
        k = 0
        for p1 in grid:
            for p2 in grid:
                distances += [np.abs(p1.R-p2.R), np.abs(p1.z - p2.z)]
            k += 1
        distances = np.asarray(distances).reshape(len(grid), len(grid), 2).astype(np.float128)
        return distances
    
    def kernel_exponential(self, thetas, distances):
        exp = thetas[0]**2 * np.exp(-.5*( (distances[:,:,0]**2)/(thetas[1]**2) + (distances[:,:,1]**2)/(thetas[2]**2))).astype(np.float128)
        return exp
    
    def marginalize_and_calculate_hps(self, sample_id, measurement, measurement_errors, inf_err_mask):
        error = measurement_errors
        measurement = measurement
        filtered_measurement = measurement[inf_err_mask]
        fL = self.geo_matrix[inf_err_mask].dot(self.prior_mean)
        bad_mll = -1e8
        best_post_mean = []
        best_post_cov = []
        best_digma_d = []
        start = time.time()
        best_sigma_fs = []
        best_sigma_xs = []
        best_sigma_xs = []
        for h, sigma_err in enumerate(self.sigma_errs):
            sigma_y = np.zeros((len(filtered_measurement), len(filtered_measurement)))
            # err_vals = (1./9)*(error[inf_err_mask]*sigma_err)**2
            err_vals = np.maximum((1./9)*(filtered_measurement*sigma_err)**2, np.ones(len(filtered_measurement)))
            np.fill_diagonal(sigma_y, err_vals)
            for i, sigma_f in enumerate(self.sigma_fs):
                for j, sigma_l1 in enumerate(self.sigma_l1s):
                    for k, sigma_l2 in enumerate(self.sigma_l2s):
                        pr_cov = self.pr_covs[i,j,k,:,:]
                        assert pr_cov.shape == (self.prior_mean.shape[0], self.prior_mean.shape[0])
                        K_ll = self.geo_matrix[inf_err_mask].dot(pr_cov).dot(self.geo_matrix_transpose[:,inf_err_mask])
                        sigma_d = (K_ll + sigma_y).astype(np.float64)
                        sigma_d_inv = np.linalg.solve(sigma_d, np.identity(len(filtered_measurement))).astype(np.float64)
        
                        mll = self.marginal_log_likelihood(y=filtered_measurement, fL=fL, cov=sigma_d, inv_cov=sigma_d_inv)
                        # print("%10.10f  %10.10f   %10.10f   %10.10f  %10.10f " % (sigma_f, sigma_l1, sigma_l2, sigma_err,  mll))
                        if mll > bad_mll:
                            
                            bad_mll = mll
                            best_hps = {u'. \u03C3err':sigma_err, u'. \u03C3f':sigma_f, u'. \u03C3l1':sigma_l1, u'. \u03C3l2':sigma_l2}
                            best_post_mean = self.prior_mean + pr_cov.dot(self.geo_matrix_transpose[:,inf_err_mask]).dot(sigma_d_inv).dot(filtered_measurement - fL)
                            best_post_cov = pr_cov - pr_cov.dot(self.geo_matrix_transpose[:,inf_err_mask]).dot(sigma_d_inv).dot(self.geo_matrix[inf_err_mask]).dot(pr_cov)
                           
                            best_sigma_d = sigma_d
                            best_sigma_f = np.zeros(len(self.sigma_fs))
                            best_sigma_l1 = np.zeros(len(self.sigma_l1s))
                            best_sigma_l2 = np.zeros(len(self.sigma_l2s))
                            best_sigma_err = np.zeros(len(self.sigma_errs))
                            best_sigma_grid = np.zeros(len(self.sigma_fs)*len(self.sigma_l1s)*len(self.sigma_l2s)*len(self.sigma_errs)).reshape(len(self.sigma_errs),
                                                                                                                                                len(self.sigma_fs),
                                                                                                                                                len(self.sigma_l1s),
                                                                                                                                                len(self.sigma_l2s))
                            best_sigma_f[i] = 1
                            best_sigma_l1[j] = 1
                            best_sigma_l2[k] = 1
                            best_sigma_err[h] = 1
                            best_sigma_grid[h, i, j, k] = 1
                            
                            
                        
        
        return best_post_mean, best_post_cov, best_hps, best_sigma_d, bad_mll, best_sigma_err, best_sigma_f, best_sigma_l1, best_sigma_l2, best_sigma_grid
    
class GaussianProcessRadial(GaussianProcessVectorial):
    def __init__(self, geo_matrix, prior_mean, grid, #boundary_matrix,
                 kernel='exponential',
                 sigma_fs = np.logspace(6,1,6),
                 sigma_l1s = np.linspace(.045, .445, 10, endpoint=False),
                 sigma_l2s = np.linspace(.045, .445, 10, endpoint=False),
                 sigma_errs = np.logspace(2, 0, 3, endpoint=True, base=2),
                 ):
        super().__init__(geo_matrix, prior_mean, grid, kernel, sigma_fs, sigma_l1s, sigma_l2s, sigma_errs)

    
    def calc_distances(self, grid):
        print('computing radial distances')
        distances = []
        k = 0
        for p1 in grid:
            for p2 in grid:
                distances += [np.abs(p1.r-p2.r), np.sin(np.abs(p1.phi - p2.phi)/2)]
            k += 1
        distances = np.asarray(distances).reshape(len(grid), len(grid), 2).astype(np.float128)
        return distances