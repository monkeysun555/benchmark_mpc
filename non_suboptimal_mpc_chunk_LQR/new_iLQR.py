import numpy as np
import math 

LQR_DEBUG = 0
iLQR_SHOW = 0
RTT_LOW = 0.02
SEG_DURATION = 1.0
CHUNK_DURATION = 0.2
CHUNK_IN_SEG = SEG_DURATION/CHUNK_DURATION
DEF_N_STEP = 5
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
MS_IN_S = 1000.0
KB_IN_MB = 1000.0
MIN_RATE = 10**-8
MAX_RATE = BITRATE[-1]/KB_IN_MB

class iLQR_solver(object):
    # Assume c(x, u) = -w1*(log(u/r0) + w2*(log(u/r0) - log(r/r0))**2 + w3*e**(-2*(b-u/w-rtt+0.2))) + 1000*e**(-10*(u+0.4))
    # Assume f(x, u) = z1 * (z2*Bu + (1-z2)*(b-u/bw-rtt+delta)) + (1-z1)*delta
    # z1 = 0/1 at 0 :  np.e**f1/(np.e**f1+1),  f1 = 100*(b-u/bw-rtt)
    # z2 = 0/1 at Bu:  np.e**f3/(np.e**f3+1),  f3 = 100*(b-u/bw-rtt+ delta - Bu)
    # And f2 = b - u/bw - rtt + delta

    def __init__(self):
        # For new traces
        self.w1 = None
        self.w2 = None
        self.w3 = 6 
        self.w4 = 3
        self.w5 = 1

        self.barrier_1 = 1
        self.barrier_2 = 1
        self.delta = 0.2  # 0.2s
        self.n_step = None
        self.predicted_bw = None
        # self.predicted_rtt = predicted_rtt
        self.predicted_rtt = None
        self.n_iteration = 50
        self.Bu = None
        self.b0 = None
        self.r0 = None
        self.target_buffer = None

    def set_target_buff(self, target):
        self.target_buffer = target

    def set_x0(self, buffer_len, rate=BITRATE[0]):
        self.b0 = np.round(buffer_len/MS_IN_S, 2)
        self.r0 = np.round(rate/KB_IN_MB, 2)
        self.target_buffer = max(min((CHUNK_IN_SEG)*self.delta, self.target_buffer), (CHUNK_IN_SEG-3)*self.delta)
        # self.target_buffer = max(self.target_buffer, (CHUNK_IN_SEG-2)*self.delta)
        if iLQR_SHOW:
            print("Initial X0 is: ", self.b0, self.r0)
            print("iLQR target buffer is: ", self.target_buffer)

    def checking(self):
        # print(self.rates[0])
        if math.isnan(self.rates[0]):
            # input() 
            self.reset()
            return True

    def nan_index(self, p_bw):
        rate_idx = 0
        for j in reversed(range(len(BITRATE))):
            if BITRATE[j]/KB_IN_MB <= p_bw:
                rate_idx = j
                break
        return rate_idx

    def reset(self):
        self.rates = [BITRATE[0]/KB_IN_MB] * self.n_step

    def set_predicted_bw_rtt(self, predicted_bw):
        assert len(predicted_bw) == self.n_step
        self.predicted_bw = [np.round(bw/KB_IN_MB, 2) for bw in predicted_bw]
        self.predicted_rtt = [RTT_LOW] * self.n_step
        if iLQR_SHOW:
            print("iLQR p_bw: ", self.predicted_bw)
            print("iLQR p_rtt: ", self.predicted_rtt)

    def set_step(self, step=DEF_N_STEP, con_type = 1):
        self.n_step = step
        self.set_weight(con_type)

    def set_weight(self, con_type):
        if con_type == 1:
            self.w1 = 1
            self.w2 = 1
        elif con_type == 2:
            self.w1 = 2
            self.w2 = 1
        elif con_type == 3:
            self.w1 = 1
            self.w2 = 1.5

    def set_bu(self, bu):
        self.Bu = bu/MS_IN_S + 1
        if iLQR_SHOW:
            print("iLQR buffer upperbound is: ", self.Bu)
            

    def set_initial_rates_trace(self, predict_trace):
        self.rates = [0.99*x /KB_IN_MB for x in predict_trace]
        self.states = []
        self.states.append([self.b0, self.r0])

    def generate_initial_x_trace(self, predict_trace):
        self.set_initial_rates_trace(predict_trace)
        for r_idx in range(len(self.rates)):
            x = self.states[r_idx]
            u = self.rates[r_idx]
            rtt = self.predicted_rtt[r_idx]
            bw = self.predicted_bw[r_idx]
            new_b = self.sim_fetch(x[0], u, rtt, bw)
            new_x = [new_b, u]
            # if r_idx < len(self.rates)-1:
            self.states.append(new_x)
        if iLQR_SHOW:
            print("iLQR rates are: ", self.rates)
            print("iLQR states are: ", self.states)

    def set_initial_rates(self, i_rate):
        self.rates = [i_rate] * self.n_step
        self.states = []
        self.states.append([self.b0, self.r0])

    def generate_initial_x(self, i_rate=BITRATE[0]/KB_IN_MB):
        self.set_initial_rates(i_rate/KB_IN_MB)
        for r_idx in range(len(self.rates)):
            x = self.states[r_idx]
            u = self.rates[r_idx]
            rtt = self.predicted_rtt[r_idx]
            bw = self.predicted_bw[r_idx]
            new_b = self.sim_fetch(x[0], u, rtt, bw)
            new_x = [new_b, u]
            # if r_idx < len(self.rates)-1:
            self.states.append(new_x)
        if iLQR_SHOW:
            print("iLQR rates are: ", self.rates)
            print("iLQR states are: ", self.states)

    def update_matrix(self, step_i):
        curr_state = self.states[step_i]
        curr_u = self.rates[step_i]
        bw = self.predicted_bw[step_i]
        rtt = self.predicted_rtt[step_i]
        b = curr_state[0]
        r = curr_state[1]
        u = curr_u
        f_1 = 100*(b-u/bw-rtt + (CHUNK_IN_SEG-1)*self.delta)
        f_2 = b-u/bw-rtt+CHUNK_IN_SEG*self.delta
        f_3 = 100*(b-u/bw-rtt + CHUNK_IN_SEG*self.delta-self.Bu)

        ce_power = -20*(b-u/bw-rtt + (CHUNK_IN_SEG-1)*self.delta + 0.05)
        ce_power_1 = -50*(u-0.1)
        ce_power_2 = 50*(u-6.5)
        ce_power_terminate = -20*(b-u/bw-rtt + CHUNK_IN_SEG*self.delta - self.target_buffer + 0.05)
        ce_buffer = b-u/bw-rtt+CHUNK_IN_SEG*self.delta-self.target_buffer
        # Without z2 for buffer upper bound
        # self.ft = [[(np.e**f_1)/(np.e**f_1+1) + (20*(np.e**f_1)*(b-1))/((np.e**f_1+1)**2), 0, -(np.e**f_1)/(bw*(np.e**f_1+1)) + (20*(u+bw)*np.e**f_1)/((np.e**f_1+1)**2)],
        #          [0, 0, 1]]

        # With z2 for buffer upper bound
        # Shape 2*3
        if LQR_DEBUG:
            print("f1 is: ", f_1)
            print("f2 is: ", f_2)
            print("b: ", b)
            print("u: ", u)
            print("rtt: ", rtt)
            print("delta: ", self.delta)
            print("bu: ", self.Bu)
            print("f3 is: ", f_3)
            input()

        # Shape 2*3
        # (b, r) = f(b', r', u) So self.ft is 2*3
        self.ft = np.array([[(100*np.e**f_1/(np.e**f_1+1)**2)*(self.Bu*np.e**f_3+f_2)/(np.e**f_3+1) + ((self.Bu*100*np.e**f_3+np.e**f_3+1-100*np.e**f_3*f_2)/(np.e**f_3+1)**2)*np.e**f_1/(np.e**f_1+1)-100*self.delta*np.e**f_1/(np.e**f_1+1)**2,
                    0, -100*np.e**f_1*(self.Bu*np.e**f_3+f_2)/(bw*(np.e**f_1+1)**2*(np.e**f_3+1)) + (np.e**f_1/(np.e**f_1+1))*(-100*self.Bu*np.e**f_3-np.e**f_3-1+100*np.e**f_3*f_2)/(bw*(np.e**f_3+1)**2) + (100*self.delta*np.e**f_1)/(bw*(np.e**f_1+1)**2)],
                   [0, 0, 1]])

        if step_i == self.n_step-1:
            # Shape 3*1
            self.ct = np.array([[-20*self.w3*np.e**ce_power-20*self.w4*np.e**ce_power_terminate, self.w2*2*np.log(r/u)/r, self.w1*-1/u + self.w2*2*np.log(u/r)/u + 20*self.w3/bw*np.e**ce_power + 20*self.w4/bw*np.e**ce_power_terminate - 50*self.barrier_1*np.e**ce_power_1 + 50*self.barrier_2*np.e**ce_power_2]]).T

            # Shape 3*3
            self.CT = np.array([[400*self.w3*np.e**ce_power+400*self.w4*np.e**ce_power_terminate, 0, -400*self.w3/bw*np.e**ce_power-400*self.w4/bw*np.e**ce_power_terminate],
                       [0, self.w2*2/(r**2)*(1-np.log(r/u)), -2*self.w2/(u*r)],
                       [-400*self.w3/bw*np.e**ce_power-400*self.w4/bw*np.e**ce_power_terminate, self.w2*-2/(u*r), self.w1/u**2 + self.w2*2/u**2*(1-np.log(u/r)) + self.w3*400*np.e**ce_power/bw**2 + self.w4*400*np.e**ce_power_terminate/bw**2 + 2500.0*self.barrier_1*np.e**ce_power_1 + 2500*self.barrier_2*np.e**ce_power_2]]).T
        else:
        # Shape 3*1
            self.ct = np.array([[-20*self.w3*np.e**ce_power, self.w2*2*np.log(r/u)/r, self.w1*-1/u + self.w2*2*np.log(u/r)/u + 20*self.w3*np.e**ce_power/bw - 50*self.barrier_1*np.e**ce_power_1 + 50*self.barrier_2*np.e**ce_power_2]]).T

            # Shape 3*3
            self.CT = np.array([[400*self.w3*np.e**ce_power, 0, -400*self.w3*np.e**ce_power/bw],
                   [0, self.w2*2/(r**2)*(1-np.log(r/u)), -2*self.w2/(u*r)],
                   [-400*self.w3*np.e**ce_power/bw, self.w2*-2/(u*r), self.w1/u**2 + self.w2*2/u**2*(1-np.log(u/r)) + self.w3*400*np.e**ce_power/bw**2 + 2500.0*self.barrier_1*np.e**ce_power_1 + 2500*self.barrier_2*np.e**ce_power_2]]).T
        
        # Add buffer cost
        # self.ct = np.array([[-20*self.w3*np.e**ce_power + 2*self.w5*ce_buffer, self.w2*2*np.log(r/u)/r, -2*ce_buffer/bw + self.w1*-1/u + self.w2*2*np.log(u/r)/u + 20*self.w3*np.e**ce_power/bw - 50*self.barrier_1*np.e**ce_power_1 + 50*self.barrier_2*np.e**ce_power_2]]).T

        # # Shape 3*3
        # self.CT = np.array([[400*self.w3*np.e**ce_power + 2*self.w5, 0, -400*self.w3*np.e**ce_power/bw - 2*self.w5/bw],
        #            [0, self.w2*2/(r**2)*(1-np.log(r/u)), -2*self.w2/(u*r)],
        #            [-400*self.w3*np.e**ce_power/bw - 2*self.w5/bw, self.w2*-2/(u*r), 2*self.w5/bw**2 + self.w1/u**2 + self.w2*2/u**2*(1-np.log(u/r)) + self.w3*400*np.e**ce_power/bw**2 + 2500.0*self.barrier_1*np.e**ce_power_1 + 2500*self.barrier_2*np.e**ce_power_2]]).T
        
        if LQR_DEBUG:
            print("Update matrix in step: ", step_i)
            print("CT matrix: ", self.CT)
            print("ct matrix: ", self.ct)
            print("ft matrix: ", self.ft)

    def iterate_LQR(self):
        # Get first loop of state using initial_u
        VT = 0
        vt = 0
        for ite_i in range(self.n_iteration):
            converge = True
            KT_list = [0.0] * self.n_step
            kt_list = [0.0] * self.n_step
            VT_list = [0.0] * self.n_step
            vt_list = [0.0] * self.n_step
            pre_xt_list = [0.0] * self.n_step
            new_xt_list = [0.0] * self.n_step
            pre_ut_list  = [0.0] * self.n_step
            d_ut_list = [0.0] * self.n_step

            # Backward pass
            for step_i in reversed(range(self.n_step)):
                self.update_matrix(step_i)
                xt = np.array([[self.states[step_i][0]],[self.states[step_i][1]]])      #2*1
                ut = np.array([[self.rates[step_i]]])                                   #1*1
                pre_xt_list[step_i] = xt
                pre_ut_list[step_i] = ut
                if step_i == self.n_step-1:
                    Qt = self.CT
                    qt = self.ct
                else:
                    # To be modified
                    Qt = self.CT + np.dot(np.dot(self.ft.T, VT), self.ft)    # 3*3
                    qt = self.ct + np.dot(self.ft.T, vt)                     # 3*1. self.ft is FT in equation, and ft in this equation is zeor (no constant)
                    if LQR_DEBUG:
                        print("vt: ", vt)
                        print("qt: ", qt)
                Q_xx = Qt[:2,:2]        #2*2
                Q_xu = Qt[:2,2].reshape((2,1))         #2*1
                Q_ux = Qt[2,:2].reshape((1,2))         #1*2
                Q_uu = Qt[2,2]                         #1*1
                q_x = qt[:2].reshape((2,1))            #2*1
                q_u = qt[2]             #1*1

                KT = np.dot(-1, np.dot(Q_uu**-1, Q_ux)).reshape((1,2))          #1*2
                kt = np.dot(-1, np.dot(Q_uu**-1, q_u))                          #1*1                
                d_u = np.dot(KT, xt) + kt                                       #1*1
                if LQR_DEBUG:
                    print("KT: ", KT)
                    print("kt: ", kt)
                    print("du: ", d_u)
                VT = Q_xx + np.dot(Q_xu, KT) + np.dot(KT.T, Q_ux) + np.dot(np.dot(KT.T, Q_uu), KT)  #2*2
                vt = q_x + np.dot(Q_xu, kt).reshape((2,1)) + np.dot(KT.T, q_u).reshape((2,1)) + np.dot(np.dot(KT.T, Q_uu), kt).reshape((2,1))    #2*1
                d_ut_list[step_i] = d_u
                KT_list[step_i] = KT
                kt_list[step_i] = kt
                VT_list[step_i] = VT
                vt_list[step_i] = vt

            # Forward pass
            new_xt_list[0] = pre_xt_list[0]
            for step_i in range(self.n_step):
                if LQR_DEBUG:
                    print("new xt: ", new_xt_list[step_i])
                    print("pre xt: ", pre_xt_list[step_i])
                    print("kt matrix is: ", kt_list[step_i])
                d_x = new_xt_list[step_i] - pre_xt_list[step_i]
                k_t = kt_list[step_i]
                K_T = KT_list[step_i]
                d_u = np.dot(K_T, d_x) + k_t

                new_u = pre_ut_list[step_i] + d_u       # New action
                if LQR_DEBUG:
                    print("New action: ", new_u)
                    input()

                # Check converge
                if converge and not np.round(new_u[0][0], 1) == np.round(self.rates[step_i],1):
                    converge = False
                self.rates[step_i] = np.round(new_u[0][0], 2)
                new_x = new_xt_list[step_i]             # Get new state
                rtt = self.predicted_rtt[step_i]
                bw = self.predicted_bw[step_i]

                new_next_b = self.sim_fetch(new_x[0][0], new_u[0][0], rtt, bw)               # Simulate to get new next state
                if step_i < self.n_step - 1:
                    new_xt_list[step_i+1] = np.array([[new_next_b], [new_u[0][0]]])
                    self.states[step_i+1] = [np.round(new_next_b, 2), self.rates[step_i]]
                    if LQR_DEBUG:
                        print("Input: ", new_x[0][0], new_u[0][0])
                        print("Output: ", new_next_b)
                        print("States: ", self.states)
                else:
                    self.states[step_i+1] = [np.round(new_next_b, 2), self.rates[step_i]]

            # Check converge
            if converge:
                break

            # ## Newly added: clip self.rates
            # self.rates = np.array(self.rates)
            # np.clip(self.rates, a_min = MIN_RATE, a_max=MAX_RATE)
            # self.rates.tolist()
            # ## Might influence system evolution

            if LQR_DEBUG:
                print("New states: ", self.states)
                print("New actions: ", self.rates)
            
            # Check convergence
            if iLQR_SHOW:
                print("Iteration ", ite_i, ", previous rate: ", self.states[0][1])
                print("Iteration ", ite_i, ", state is: ", [x[0] for x in self.states])
                print("Iteration ", ite_i, ", pre bw is: ", self.predicted_bw)
                print("Iteration ", ite_i, ", action is: ", self.rates)
                print("<===============================================>")

        # r_idx = self.translate_to_rate_idx()
        # return r_idx
        return self.rates[0]

    def get_rates(self):
        return self.rates

    def translate_to_rate_idx(self):
        first_action = self.rates[0]
        # distance = [np.abs(first_action-br/KB_IN_MB) for br in BITRATE]
        # rate_idx = distance.index(min(distance))
        rate_idx = 0
        for j in reversed(range(len(BITRATE))):
            if BITRATE[j]/KB_IN_MB <= first_action:
                rate_idx = j
                break
        if iLQR_SHOW:
            print("Rate is: ", first_action)
            print("Rate index: ", rate_idx)
            # input()
        return rate_idx

    def sim_fetch(self, buffer_len, seg_rate, rtt, bw, state = 1, playing_speed = 1.0):
        seg_size = seg_rate
        # print('Seg size is: ', seg_size)
        # Chunk downloading
        freezing = 0.0
        wait_time = 0.0
        current_reward = 0.0
        download_time = seg_size/bw + rtt

        freezing = max(0.0, download_time - buffer_len - (CHUNK_IN_SEG-1)*self.delta)
        buffer_len = max(buffer_len - download_time + (CHUNK_IN_SEG-1)*self.delta, 0.0)
        buffer_len += self.delta
        if freezing > 0.0:
            assert buffer_len == self.delta
        buffer_len = min(self.Bu, buffer_len)
        return buffer_len 

    # def LQR(self, step_i):
    #     xt = np.array([[self.states[step_i][0]],[self.states[step_i][1]]])  #2*1
    #     ut = np.array([[self.us[step_i]]])                                  #1*1

    #     if step_i == self.n_step-1:
    #         Qt = self.CT
    #         qt = self.ct
    #     else:
    #         # To be modified
    #         Qt = self.CT + self.ft.T


    #     Q_xx = Qt[:2,:2]        #2*2
    #     Q_xu = Qt[:2,2]         #2*1
    #     Q_ux = Qt[2,:2]         #1*2
    #     Q_uu = Qt[2,2]          #1*1
    #     q_x = qt[:2]            #2*1
    #     q_u = qt[2]             #1*1


    #     KT = np.dot(-1, np.dot(Q_uu**-1, Q_ux))         #1*2
    #     kt = np.dot(-1, np.dot(Q_uu**-1, q_u))          #1*1
    #     d_u = np.dot(KT, xt) + kt
    #     VT = Q_xx + np.dot(Q_xu, KT) + np.dot(KT.T, Q_ux) + np.dot(np.dot(KT.T, Q_uu), KT)  #2*2
    #     vt = q_x + np.dot(Q_xu, kt) + np.dot(KT.T, q_u) + np.dot(np.dot(KT.T, Q_uu), kt)

