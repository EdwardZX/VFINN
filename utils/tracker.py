import numpy as np 
from utils.kalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque


class Tracks(object):
	"""docstring for Tracks"""
	def __init__(self, detection, trackId, state_dim = 4, deque_max = 5000):
		super(Tracks, self).__init__()
		self.KF = KalmanFilter()
		self.KF.predict()
		self.KF.correct(np.matrix(detection).reshape(state_dim,1))
		self.trace = deque(maxlen = deque_max)
		self.prediction = detection.reshape(1, state_dim)
		self.observation = detection.reshape(1, state_dim)
		self.trackId = trackId
		self.skipped_frames = 0
		self.state_dim = state_dim

	def predict(self,detection):
		self.prediction = np.array(self.KF.predict()).reshape(1,self.state_dim)
		self.KF.correct(np.matrix(detection).reshape(self.state_dim,1))
		self.observation = detection.reshape(1,self.state_dim)


class TracksLAP(object):
	"""docstring for Tracks"""
	def __init__(self, detection, trackId, state_dim = 4, deque_max = 5000):
		super(TracksLAP, self).__init__()
		self.trace = deque(maxlen = deque_max)
		self.prediction = detection.reshape(1,state_dim)
		self.observation = detection.reshape(1, state_dim)
		self.trackId = trackId
		self.skipped_frames = 0
		self.state_dim = state_dim

	def predict(self,detection):
		tmp = detection.copy()
		# if tmp.shape[0] == 4: # the pos point
		tmp[0] = tmp[0] + tmp[1]
		tmp[2] = tmp[2] + tmp[3]
		self.prediction = tmp.reshape(1, self.state_dim)
		self.observation = detection.reshape(1, self.state_dim)


class Tracker(object):
	"""docstring for Tracker"""
	def __init__(self, dist_threshold, max_frame_skipped,
				 max_trace_length = 5000, track_method = 'kalman',
				 cost_const = 0.1):
		super(Tracker, self).__init__()
		self.dist_threshold = dist_threshold
		self.max_frame_skipped = max_frame_skipped
		self.max_trace_length = max_trace_length
		self.trackId = 0
		self.tracks = []
		self.track_method = track_method
		self.cost_const = cost_const
		# self.state_dim = state_dim

	def update_with_frame(self, detections, frame_id):
		if len(self.tracks) == 0:
			for i in range(detections.shape[0]):
				if self.track_method == 'lap':
					track = TracksLAP(detections[i], self.trackId)
				else:
					track = Tracks(detections[i], self.trackId)
				self.trackId +=1
				self.tracks.append(track)

		N = len(self.tracks)
		M = len(detections)

		# select frame less than
		cost = []
		cost_raw = []
		for i in range(N):
			# diff = np.linalg.norm(self.tracks[i].prediction - detections.reshape(-1,2), axis=1)
			state_dim = self.tracks[i].prediction.shape[-1]
			y_pred = self.tracks[i].prediction.reshape(-1,state_dim)[:, [0, 2]] #pos
			y_obs = self.tracks[i].observation.reshape(-1,state_dim)[:, [0, 2]] #pos

			y_detect = detections.reshape(-1,state_dim)[:, [0, 2]]#pos
			diff_pred = np.linalg.norm(y_pred - y_detect, axis = 1)
			diff_raw = np.linalg.norm(y_obs - y_detect, axis=1)


			diff = np.minimum(diff_pred, diff_raw)
			cost.append(diff)
			cost_raw.append(diff_raw)

		cost = np.array(cost)*self.cost_const # help the numeric range
		cost_raw = np.array(cost_raw) * self.cost_const  # help the numeric range
		row, col = linear_sum_assignment(cost)
		assignment = [-1]*N
		for i in range(len(row)):
			assignment[row[i]] = col[i]

		un_assigned_tracks = []

		for i in range(len(assignment)):
			self.tracks[i].skipped_frames += 1
			if assignment[i] != -1:
				if (cost[i][assignment[i]] > self.dist_threshold*self.cost_const)\
						or (cost_raw[i][assignment[i]] > self.dist_threshold*self.cost_const*2e1):
					assignment[i] = -1
					un_assigned_tracks.append(i)
				# else:
				# 	self.tracks[i].skipped_frames = 0


		for i in range(len(detections)):
			if (i not in assignment):# or (self.tracks[i].skipped_frames > self.max_frame_skipped):
				track = Tracks(detections[i], self.trackId)
				self.trackId +=1
				self.tracks.append(track)



		for i in range(len(assignment)):
			# not considering the single point
			if (assignment[i] != -1): #and (self.tracks[i].skipped_frames <= self.max_frame_skipped):
				self.tracks[i].skipped_frames = 0

				self.tracks[i].predict(detections[assignment[i]])
			# self.tracks[i].trace.append(self.tracks[i].prediction)

				t_pos = np.array([frame_id,self.tracks[i].observation[0,0],self.tracks[i].observation[0,2]])
				self.tracks[i].trace.append(t_pos)









		



