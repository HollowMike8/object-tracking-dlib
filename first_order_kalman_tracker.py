import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class KalmanBoxTracker(object):
    '''
    This class represents state of individual tracked objects observed as bbox.
    '''
    
    def _convert_bbox_to_z(self, bbox):
        '''
        Converts bbox parameters (top left and bottom right corners) 
        [x1,y1,x2,y2] into [x,y,s,r] where x,y is the centre of the bbox 
        and s is the area and r isthe aspect ratio
        '''
        w = bbox[2]-bbox[0]
        h = bbox[3]-bbox[1]
        x = bbox[0]+w/2.
        y = bbox[1]+h/2.
        s = w*h
        r = w/float(h)        
        return np.array([x,y,s,r]).reshape((4,1))
    
    def _convert_x_to_bbox(self, x, score=None):
      """
      Takes a bounding box in the centre form [x,y,s,r] and returns it in the 
      form [x1,y1,x2,y2] where x1,y1 is top left and x2,y2 is bottom right
      """
      w = np.sqrt(x[2]*x[3])
      h = x[2]/w
      
      bbox = np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
      
      if(score==None):
          return bbox
      else:
        return bbox, score
    
    def __init__(self, bbox, dt=1, img=None):
        '''
        Intialize the kalman tracker with the given bounding box

        Parameters
        ----------
        bbox : list
            Bounding box parameters (top left and bottom right corners).
        dt : int
            Time step. The default is 1.
        img : ndarray, optional
            Image in which the objects are tracked. The default is None.

        Returns
        -------
        None.

        '''
        # use constant velocity model (A*X + w)
        # X:[x, x_cap, y, y_cap] and Z:[x, y]
        
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # state transition matrix
        self.kf.F = np.array([[1, dt, 0,  0],
                              [0,  1, 0,  0],
                              [0,  0, 1, dt],
                              [0,  0, 0,  1]])
        
        # process noise matrix (discrete) --> assumption:x, y are independent
        q = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
        self.kf.Q = block_diag(q, q)
        
        # control matrix --> default = None
        self.kf.B = None
        
        # measurement matrix --> Z = H*X
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0]])
        
        # measurement noise matrix --> variance of x, y in pixel^2
        self.kf.R = np.array([[10, 0],
                              [0, 10]])
        
        # initial conditions
        self.kf.P = np.eye(4)*500
        self.kf.x = self._convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def update(self, bbox, img=None):
        '''
        Updates the state vector with observed bbox.
        '''
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        if bbox != []:
            self.kf.update(self._convert_bbox_to_z(bbox))

    def predict(self, img=None):
        '''
        Advances the state vector and returns the predicted bbox estimate.
        '''
        self.kf.predict()
        
        self.age += 1
        if(self.time_since_update > 0):
          self.hit_streak = 0
        self.time_since_update += 1
        
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        
        return self.history[-1][0]

    def get_state(self):
        '''
        Returns the current bounding box estimate.
        '''
        return self._convert_x_to_bbox(self.kf.x)[0]