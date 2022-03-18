import numpy as np
import dlib

class CorrelationTracker(object):
    '''
    This class represents state of individual tracked objects observed as bbox.
    ''' 
    def __init__(self, bbox, img):
        '''
        Intialize the correlation tracker with the given bounding box

        Parameters
        ----------
        bbox : list
            Bounding box parameters (top left and bottom right corners).
        img : ndarray
            Image in which the objects are tracked.

        Returns
        -------
        None.

        '''    
        self.tracker = dlib.correlation_tracker()
        self.rect = dlib.rectangle(int(bbox[0]), int(bbox[1]), 
                                   int(bbox[2]), int(bbox[3]))
                                   
        self.tracker.start_track(img, self.rect)
        
        self.confidence = 0.
        self.time_since_update = 0
        CorrelationTracker.count = 0
        self.id = CorrelationTracker.count
        CorrelationTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox, img):
        '''
        Restarts the tracker with the observed bbox.
        '''
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
    
        if bbox != []:
            self.rect = dlib.rectangle(int(bbox[0]), int(bbox[1]), 
                                       int(bbox[2]), int(bbox[3]))
            self.tracker.start_track(img, self.rect)

    def predict(self, img):
        '''
        Advances the state vector and returns the predicted bbox estimate.
        '''
        self.confidence = self.tracker.update(img)
        
        self.age += 1
        if (self.time_since_update > 0):
          self.hit_streak = 0
        self.time_since_update += 1
        
        return self.get_state()

    def get_state(self):
        '''
        Returns the current bounding box estimate.
        '''
        pos = self.tracker.get_position()
        
        return [int(pos.left()), int(pos.top()), int(pos.right()), 
                int(pos.bottom())]
