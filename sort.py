import numpy as np
from correlation_tracker import CorrelationTracker
from first_order_kalman_tracker import KalmanBoxTracker
from data_association import associate_detections_to_trackers

class Sort(object):
    
    def __init__(self, max_age=1, min_hits=3, use_dlib = False):
        '''
        Initialize the key parameters for SORT.

        Parameters
        ----------
        max_age : int, optional
            No. of consecutive unmatched detections before a track is deleted. 
            The default is 1.
        min_hits : int, optional
            No. of consecutive matches needed to establish track. Default is 3.
        use_dlib : boolean, optional
            Whether to use correlation tracker or not. The default is False.

        Returns
        -------
        None.

        '''
        self.max_age = max_age
        self.min_hits = min_hits
        self.use_dlib = use_dlib
        
        self.trackers = []
        self.frame_count = 0
        
    def update(self, dets, img=None):
        '''
        Parameters
        ----------
        dets : array
            Detections in the format [[x1,y1,x2,y2],[x1,y1,x2,y2],...].
        img : ndarray, optional
            Image in which the objects are tracked. The default is None.

        Returns
        -------
        Array similar to detection but with object ID concatenated at the end.

        '''
        self.frame_count += 1
        
        # initialize lists to delete and retain trackers
        to_del = []
        ret = []
        
        # get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers),5))
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict(img)
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if(np.any(np.isnan(pos))):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        if dets != []:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)
              
        # update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
            if(t not in unmatched_trks):
                d = matched[np.where(matched[:,1]==t)[0],0]
                trk.update(dets[d,:][0], img)   
        
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            if not self.use_dlib:
                trk = KalmanBoxTracker(dets[i,:])
            else:
                trk = CorrelationTracker(dets[i,:], img)
            self.trackers.append(trk)        
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if dets == []:
                trk.update([],img)
            d = trk.get_state()
            
            if((trk.time_since_update < 1) and 
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
            
            i -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                
        if(len(ret) > 0):
            return np.concatenate(ret)
        else:
            return np.empty((0,5))