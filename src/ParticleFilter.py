import os
import cv2
import numpy as np
# import glob

class State:
    """ 
    @brief          Initialize the state of the particle.
    @param x        x coordinate of the center of the ellipses
    @param y        y coordinate of the center of the ellipses
    @param Hx       length of half axis along x
    @param Hy       length of half axis along y
    @param x_dot    motion along x (default 0.0)
    @param y_dot    motion along y (default 0.0)
    @param a_dot    corresponding scale change (default 0.0)
    """
    def __init__(self, x, y, Hx, Hy, x_dot=0., y_dot=0., a_dot=0.):
        self.x = x
        self.y = y
        self.Hx = Hx
        self.Hy = Hy
        self.x_dot = x_dot
        self.y_dot = y_dot
        self.a_dot = a_dot

    def draw_point(self, frame, path=None):
        cv2.circle(frame, center=(self.x, self.y), radius=1, color=(255, 0, 255), thickness=1)
        if path is not None:
            cv2.imwrite(path, frame)
        return frame
    
    def draw_box(self, frame, path=None):
        cv2.rectangle(frame, (self.x - self.Hx, self.y - self.Hy), ( self.x + self.Hx, self.y + self.Hy), (0, 0, 255),thickness=2)
        if path is not None:
            cv2.imwrite(path, frame)
        return frame
    
    def __str__(self):
        txt = f'x: {self.x}, y: {self.y}, Hx: {self.Hx}, Hy: {self.Hy}'
        return txt

class Histogram:
    def __init__(self, size=8, max_range=255.0):
        self.size = size
        self.max_range = max_range
        self.bins = [max_range/size * i for i in range(size)]
        self.height = np.array([0.0 for i in range(size)])
    
    def update(self, idx):
        self.height[idx] =+ 1

    def get_histogram_idx(self, val):
        for i in range(self.size - 1):
            if val >= self.bins[i] and val < self.bins[i+1]:
                return i
            if val >= self.bins[-1] and val <= self.max_range:
                return self.size - 1
        print(val)
        print(self.bins)

class ParticleFilter:
    DELTA_T = 0.05
    VELOCITY_DISTURB = 4.0
    SCALE_DISTURB = 0.0
    SCALE_CHANGE_D = 0.001

    def __init__(self, frames, init_x, init_y, init_Hx, init_Hy,
    particle_count=50, out_path='../output'):
        self.frames = frames
        self.f_idx = 0
        self.particle_count = particle_count
        self.particles = []
        self.out_path = out_path

        init_state = State(x=init_x, y=init_y, Hy=init_Hy, Hx=init_Hx)
        init_state.draw_point(frames[self.f_idx], path=os.path.join(out_path, f'{self.f_idx}_points.jpg'))
        init_state.draw_box(frames[self.f_idx], path=os.path.join(out_path, f'{self.f_idx}.jpg'))
        self.state = init_state
        random_list = np.random.normal(scale=0.4, size=(particle_count, 7))
        self.weights = [1.0 / particle_count] * particle_count

        for i in range(particle_count):
            x0 = int(init_state.x + random_list[i, 0] * init_state.Hx)
            y0 = int(init_state.y + random_list[i, 1] * init_state.Hy)
            x0_H = int(init_state.Hx + random_list[i, 4] * self.SCALE_DISTURB)
            y0_H = int(init_state.Hy + random_list[i, 5] * self.SCALE_DISTURB)
            x0_d = init_state.x_dot + random_list[i, 2] * self.VELOCITY_DISTURB
            y0_d = init_state.y_dot + random_list[i, 3] * self.VELOCITY_DISTURB
            a0_d = init_state.a_dot + random_list[i, 6] * self.SCALE_CHANGE_D

            particle = State(x0, y0, x0_H, y0_H, x0_d, y0_d, a0_d)
            particle.draw_point(frame=frames[self.f_idx], path=self.out_path+'/%04d.jpg'%(self.f_idx+1))
            self.particles.append(particle)

        self.q = [Histogram(size=2, max_range=180.0), Histogram(size=2, max_range=255.0), Histogram(size=10, max_range=255.0)]
        first_frame = self.frames[0]
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)

        for hist in self.q:
            for u in range(hist.size):
                a = np.sqrt(init_state.Hx ** 2 + init_state.Hy ** 2)
                f = 0
                weights = []
                x_bins = []
                for i in range(init_state.x - init_state.Hx, init_state.x + init_state.Hx):
                    for j in range(init_state.y - init_state.Hy, init_state.y + init_state.Hy):
                        x_val = first_frame[j][i][self.q.index(hist)]
                        temp = self.calc_k(np.linalg.norm((j - init_state.y, i - init_state.x)) / a)
                        weights.append(temp)
                        f += temp
                        x_bins.append(self.kronecker_delta(hist.get_histogram_idx(float(x_val)) - u))
                hist.height[u] = np.sum(np.array(weights) * np.array(x_bins)) / f
    
    def select(self):
        """ 
        @brief Selects N samples from set S[t-1] with some probability π[t-1]. 
        @return Returns -1 if f_idx is out of range.
        """

        if self.f_idx < len(self.frames) - 1:
            self.f_idx += 1
        else:
            return -1
        print(f'selection on frame {self.f_idx}')
        self.frame = self.frames[self.f_idx]
        indices = self.get_random_index(self.weights)
        new_particles = []
        for i in indices:
            new_particles.append(State(
                x=self.particles[i].x, y=self.particles[i].y,
                Hx=self.particles[i].Hx, Hy=self.particles[i].Hy,
                x_dot=self.particles[i].x_dot, y_dot=self.particles[i].y_dot, a_dot=self.particles[i].a_dot
                ))
        self.particles = new_particles
        return 0
    
    def propagate(self):
        """ 
        @brief Propagates each sample from set S'[t-1] 
        by a linear stochastic differential equation: 
        S[t] = A.S[t-1] + w[t-1] 
        """
        temp = []
        for particle in self.particles:
            random_nums = np.random.normal(0, 0.4, 7)
            particle.x = int(particle.x+particle.x_dot*self.DELTA_T+random_nums[0]*particle.Hx+0.5)
            particle.y = int(particle.y+particle.y_dot*self.DELTA_T+random_nums[1]*particle.Hy+0.5)
            particle.x_dot = particle.x_dot+random_nums[2]*self.VELOCITY_DISTURB
            particle.y_dot = particle.y_dot+random_nums[3]*self.VELOCITY_DISTURB
            particle.Hx = int(particle.Hx*(particle.a_dot+1)+random_nums[4]*self.SCALE_DISTURB+0.5)
            particle.Hy = int(particle.Hy*(particle.a_dot+1)+random_nums[5]*self.SCALE_DISTURB+0.5)
            particle.a_dot = particle.a_dot+random_nums[6]*self.SCALE_CHANGE_D
            temp.append(particle.draw_point(self.frames[self.f_idx], self.out_path+'/%04d.jpg'%(self.f_idx+1)))
        return temp
    
    def observe(self):
        """ @brief Observes the colour distributions and weights them. """
        frame = self.frames[self.f_idx]
        height, width = frame.shape[:2]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Bhattacharyya coeffient
        B = []
        for i in range(self.particle_count):
            if self.particles[i].x < 0 or self.particles[i].x > width - 1 \
            or self.particles[i].y < 0 or self.particles[i].y > height - 1:
                B.append(0)
                continue
            
            # based on cv2 HSV ranges
            self.p = [Histogram(size=2, max_range=180.0), Histogram(size=2, max_range=255.0), Histogram(size=10, max_range=255.0)]
            for hist in self.p:
                for u in range(hist.size):
                    # region size adaptation
                    a = np.sqrt(self.particles[i].Hx ** 2 + self.particles[i].Hy ** 2)
                    # normalization factor
                    f = 0
                    weights = []
                    x_bins = []
                    for m in range(self.particles[i].x - self.particles[i].Hx, self.particles[i].x + self.particles[i].Hx):
                        for n in range(self.particles[i].y - self.particles[i].Hy, self.particles[i].y + self.particles[i].Hy):
                            if n >= height:
                                n = height - 1
                            elif n < 0:
                                n=0
                            if m >= width:
                                m = width - 1
                            elif m < 0:
                                m = 0
                            x_val = frame[n][m][self.p.index(hist)]
                            temp = self.calc_k(np.linalg.norm((m - self.particles[i].x, n - self.particles[i].y)) / a)
                            f += temp
                            x_bins.append(self.kronecker_delta(hist.get_histogram_idx(x_val) - u))
                            weights.append(temp)
                    hist.height[u] = np.sum(np.array(weights) * np.array(x_bins)) / f
            B_temp = self.B_coefficient(
                np.concatenate((self.p[0].height, self.p[1].height, self.p[2].height)),
                np.concatenate((self.q[0].height, self.q[1].height, self.q[2].height))
                )
            B.append(B_temp)
        for i in range(self.particle_count):
            self.weights[i] = self.get_weight(B[i])
        self.weights /= sum(self.weights)
    
    def estimate(self):
        """ @brief Estimates the mean state of S[t]. """
        self.state.x = np.sum(np.array([s.x for s in self.particles]) * self.weights).astype(int)
        self.state.y = np.sum(np.array([s.y for s in self.particles]) * self.weights).astype(int)
        self.state.Hx = np.sum(np.array([s.Hx for s in self.particles]) * self.weights).astype(int)
        self.state.Hy = np.sum(np.array([s.Hy for s in self.particles]) * self.weights).astype(int)
        self.state.x_dot = np.sum(np.array([s.x_dot for s in self.particles]) * self.weights)
        self.state.y_dot = np.sum(np.array([s.y_dot for s in self.particles]) * self.weights)
        self.state.a_dot = np.sum(np.array([s.a_dot for s in self.particles]) * self.weights)
        print(self.state)
        temp = self.state.draw_box(self.frames[self.f_idx], self.out_path+'/%04d.jpg'%(self.f_idx + 1))
        return temp
    
    def calc_k(self, r):
        """ 
        @brief      Weighting function for pixels.
        @param r    distance from the pixel to the centre of ellipse.
        """
        if r < 1:
            return 1 - (r ** 2)
        return 0
    
    def kronecker_delta(self, x):
        """
        @brief      Kronecker delta function.
        @param x    index of the histogram.
        """
        if abs(x) < 0.1:
            return 1
        return 0

    def B_coefficient(self, p, q):
        """ 
        @brief      Calculate Bhattacharyya coefficient.
        @param p    histogram of the first image.
        @param q    histogram of the second image.
        """
        return np.sum(np.sqrt(p * q))

    def get_weight(self, B):
        """ 
        @brief      Weighting function
        @param B    Bhattacharyya coefficient.
        """
        return np.exp(-(1.-B) / 0.02)
    
    def get_random_index(self, weights):
        weights_acc = []
        idx = []
        weights_acc.append(0)
        for i in range(len(weights)):
            weights_acc.append(weights_acc[i] + weights[i])
        for i in range(len(weights)):
            r = np.random.rand()
            for j in range(len(weights)):
                if r > weights_acc[j] and r < weights_acc[j+1]:
                    idx.append(j)
                    continue
        return idx


        

