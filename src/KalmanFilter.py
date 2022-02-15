# -*- coding: utf-8 -*-

import numpy as np


class KalmanFilter:
    def __init__(self, xk, Pk, A, B, Q, H, R):
        """
        @brief          Manual Initialization
        @param xk:      state estimate
        @param Pk:      covariance matrix
        @param A:       transition/prediction step matrix (Fk)
        @param B:       control matrix
        @param Q:       uncertainty/noise covariance
        @param H:       sensor matrix
        @param R:       sensor noise covariance
        """
        self.xk = xk
        self.Pk = Pk
        self.A = A
        self.B = B
        self.Q = Q
        self.H = H
        self.R = R

    def __init__(self):
        dt = 1
        # state estimate
        self.xk = np.zeros((4, 1))
        # covariance matrix
        self.Pk = np.eye(4)
        # transition/prediction step matrix
        self.A = np.array([
            [1,  0, dt,  0],
            [0,  1,  0, dt],
            [0,  0,  1,  0],
            [0,  0,  0,  1]
        ])
        # noise/uncertainty covariance
        self.Q  = np.eye(4)
        # sensor model
        self.H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
        # sensor noise
        self.R = np.array([
                [1, 0],
                [0, 1]
            ]) 
        # control matrix
        self.B = np.array([
            [dt ** 2 / 2, 0],
            [0, dt ** 2 / 2],
            [dt, 0],
            [0, dt]
        ])

    def predict(self):
        u = np.array([0.001, 0.001])
        # state estimate X
        self.xk = np.dot(self.A, self.xk) + np.dot(self.B, u)
        self.Pk = np.dot(np.dot(self.A, self.Pk), self.A.T) + self.Q
        return self.xk

    def correct(self, zk):
        """ 
        @brief  Measurement Update phase
        @param  zk: measurement
        @return xk: state estimate
        """
        n = self.A.shape[1]
        # temp = H * Pk * H.T + R
        temp = np.dot(self.H, np.dot(self.Pk, self.H.T)) + self.R
        # K' = Pk * H.T * inv(H * Pk * H.T + R)
        kalman_gain = np.dot(
                np.dot(self.Pk, self.H.T),
                np.linalg.inv(temp)
            )
        
        # y = zk - H * xk
        y = zk - np.dot(self.H, self.xk)
        # x'k = xk + K * (zk - H * xk)
        self.xk = self.xk + np.dot(kalman_gain, y)

        # P'k = Pk - (K * H * Pk)
        self.Pk = self.Pk - np.dot(np.dot(kalman_gain, self.H), self.Pk)
        return self.xk
    