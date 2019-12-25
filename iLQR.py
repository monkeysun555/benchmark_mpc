class iLQR(object):
    # Assume c(x, u) = -w1*(log(u/r0) - w2*(log(u/r0) - log(r/r0))**2 - w3*(0.5*e**(-2*(b-u/w-rtt+1))))
    # Assume f(x, u) = z1 * (z2*Bu + (1-z2)*(b-u/bw-rtt+delta)) + (1-z1)
    # z1 = 0/1 at 0 :  np.e**f1/(np.e**f1+1),  f1 = 20*(b-u/bw-rtt)
    # z2 = 0/1 at Bu:  np.e**f3/(np.e**f3+1),  f3 = 20*(b-u/bw-rtt+ delta - Bu)
    # And f2 = b - u/bw - rtt + delta
    def __init__(self, predicted_bw, predicted_rtt, n_step):
        self.w1 = 1
        self.w2 = 1
        self.w3 = 6     
        self.delta = 1  # 1s
        self.initial_u = [0.3 for i in range(len(predict_bw))]
        self.predicted_bw = predicted_bw
        self.predicted_rtt = predicted_rtt
        self.n_iteration = 50
        self.Bu = None
        self.n_step = n_step

    def update_matrix(self, step_i):
        curr_state = self.states[step_i]
        curr_u = self.us[step_i]
        bw = self.predicted_bw[step_i]
        rtt = self.predicted_rtt[step_i]
        b = curr_state[0]
        r = curr_state[1]
        u = curr_u
        f_1 = 20*(b-u/bw-rtt)
        f_2 = b-u/bw-rtt+self.delta
        f_3 = 20*(b-u/bw-rtt+delta-self.Bu)
        ce_power = -2*(b-u/bw-rtt+1)

        # Without z2 for buffer upper bound
        # self.ft = [[(np.e**f_1)/(np.e**f_1+1) + (20*(np.e**f_1)*(b-1))/((np.e**f_1+1)**2), 0, -(np.e**f_1)/(bw*(np.e**f_1+1)) + (20*(u+bw)*np.e**f_1)/((np.e**f_1+1)**2)],
        #          [0, 0, 1]]

        # With z2 for buffer upper bound
        # Shape 2*3
        self.ft = np.array([[(20*np.e**f_1/((np.e**f_1+1)**2))*((self.Bu*np.e**f_3+ f_2)/(np.e**f_3+1)) + ((self.Bu*20*np.e**f_3+np.e**f_3+1-20*np.e**f_3*f_2)/(np.e**f_3+1)**2)*np.e**f_1/(np.e**2+1)-20*np.e**f_1/(np.e**f_1+1)**2,
                    0, -20*np.e**f_1*(self.Bu*np.e**f_3+f_2)/(bw*(np.e**f_1+1)**2*(np.e**f_3+1)) + (np.e**f_1/(np.e**f_1+1))*(-20*self.Bu*np.e**f_3-np.e*f_3-1+20*np.e**f_3*f_2)/(bw*(np.e**f_3+1)**2) + (20*np.e**f_1)/(bw*(np.e**f_2+1)**2)],
                   [0, 0, 1]])

        # Shape 3*1
        self.ct = np.array([[-self.w3*np.e**ce_power, (2*np.log(r/u))/r, 1/u + (2*np.log(u/r))/u+(self.w3/bw)*np.e**ce_power]]).T

        # Shape 3*3
        self.CT = np.array([[2*self.w3*np.e**ce_power, 0, (-2*self.w3/bw)*np.e**ce_power],
                   [0, 2/(r**2)*(1-np.log(r/u)), -2/(u*r)],
                   [(-2*self.w3*(np.e**ce_power))/bw, -2/(u*r), -(u**2)+(2/(u**2))*(1-np.log(u/r)) + (2*np.e**ce_power)/(bw**2)]]).T

    def iterate_LQR(self):
        # Get first loop of state using initial_u
        VT = 0
        vt = 0
        for i in range(self.n_iteration):
            KT_list = [0.0] * self.n_step
            kt_list = [0.0] * self.n_step
            VT_list = [0.0] * self.n_step
            vt_list = [0.0] * self.n_step
            pre_xt_list = [0.0] * self.n_step
            new_xt_list = [0.0] * self.n_step
            pre_ut_list  = [0.0] * self.n_step
            new_ut_list = [0.0] * self.n_step
            d_ut_list = [0.0] * self.n_step

            # Simulate for steps

            # Backward pass
            for step_i in reversed(range(self.n_step)):
                xt = np.array([[self.states[step_i][0]],[self.states[step_i][1]]])  #2*1
                ut = np.array([[self.us[step_i]]])                                  #1*1
                pre_xt_list[step_i] = xt
                pre_ut_list[step_i] = ut
                if step_i == self.n_step-1:
                    Qt = self.CT
                    qt = self.ct
                else:
                    # To be modified
                    Qt = self.CT + np.dot(np.dot(self.ft.T, VT), self.ft)    # 3*3
                    qt = self.ct + np.dot(self.ft.T, vt)                     # 3*1. self.ft is FT in equation, and ft in this equation is zeor (no constant)
                Q_xx = Qt[:2,:2]        #2*2
                Q_xu = Qt[:2,2]         #2*1
                Q_ux = Qt[2,:2]         #1*2
                Q_uu = Qt[2,2]          #1*1
                q_x = qt[:2]            #2*1
                q_u = qt[2]             #1*1

                KT = np.dot(-1, np.dot(Q_uu**-1, Q_ux))         #1*2
                kt = np.dot(-1, np.dot(Q_uu**-1, q_u))          #1*1
                d_u = np.dot(KT, xt) + kt
                VT = Q_xx + np.dot(Q_xu, KT) + np.dot(KT.T, Q_ux) + np.dot(np.dot(KT.T, Q_uu), KT)  #2*2
                vt = q_x + np.dot(Q_xu, kt) + np.dot(KT.T, q_u) + np.dot(np.dot(KT.T, Q_uu), kt)    #2*1

                d_ut_list[step_i] = d_u
                KT_list[step_i] = KT
                kt_list[step_i] = kt
                VT_list[step_i] = VT
                vt_list[step_i] = vt

            # Forward pass
            new_xt_list[0] = pre_xt_list[0]
            for step_i in range(self.n_step):
                d_x = new_xt_list[step_i] - pre_xt_list[step_i]
                k_t = kt_list[step_i]
                K_T = KT_list[step_i]
                d_u = np.dot(K_T, d_x) + k_t
                new_u = pre_ut_list[step_i] + d_u
                pre_x = pre_xt_list[step_i]
                new_x =  f(x)               # Simulate to get new state
                new_xt_list[step_i+1] = new_x

    def LQR(self, step_i):
        xt = np.array([[self.states[step_i][0]],[self.states[step_i][1]]])  #2*1
        ut = np.array([[self.us[step_i]]])                                  #1*1

        if step_i == self.n_step-1:
            Qt = self.CT
            qt = self.ct
        else:
            # To be modified
            Qt = self.CT + self.ft.T


        Q_xx = Qt[:2,:2]        #2*2
        Q_xu = Qt[:2,2]         #2*1
        Q_ux = Qt[2,:2]         #1*2
        Q_uu = Qt[2,2]          #1*1
        q_x = qt[:2]            #2*1
        q_u = qt[2]             #1*1


        KT = np.dot(-1, np.dot(Q_uu**-1, Q_ux))         #1*2
        kt = np.dot(-1, np.dot(Q_uu**-1, q_u))          #1*1
        d_u = np.dot(KT, xt) + kt
        VT = Q_xx + np.dot(Q_xu, KT) + np.dot(KT.T, Q_ux) + np.dot(np.dot(KT.T, Q_uu), KT)  #2*2
        vt = q_x + np.dot(Q_xu, kt) + np.dot(KT.T, q_u) + np.dot(np.dot(KT.T, Q_uu), kt)

    def sim_fetching(self):
        # Action initialization
        # print "start fetching, seg idx is:", seg_idx
        start_state = self.state
        chunk_size = next_chunk_set # in Kbits not KBytes
        chunk_start_time = seg_idx * self.seg_duration + chunk_idx * self.chunk_duration
        # as mpd is based on prediction, there is noise
        # chunk_size = np.random.uniform(CHUNK_RANDOM_RATIO_LOW*chunk_size, CHUNK_RANDOM_RATIO_HIGH*chunk_size)
        chunk_sent = 0.0    # in Kbits
        downloading_fraction = 0.0  # in ms
        freezing_fraction = 0.0 # in ms
        time_out = 0
        # rtt = 0.0
        # Handle RTT 
        if take_action:
            rtt = np.random.uniform(RTT_LOW, RTT_HIGH)  # in ms
            # rtt = RTT_LOW # For upper bound calculation
            duration = self.time_trace[self.time_idx] * MS_IN_S - self.last_trace_time  # in ms
            if duration > rtt:
                self.last_trace_time += rtt
            else:
                temp_rtt = rtt - duration
                self.last_trace_time = self.time_trace[self.time_idx] * MS_IN_S # in ms
                self.time_idx += 1
                if self.time_idx >= len(self.time_trace):
                    self.time_idx = 1
                    self.last_trace_time = 0.0
                self.last_trace_time += temp_rtt
                assert self.last_trace_time < self.time_trace[self.time_idx] * MS_IN_S
            downloading_fraction += rtt
            assert self.state == 1 or self.state == 0
            # Check whether during startup
            if self.state == 1:
                self.playing_time += np.minimum(self.buffer, playing_speed*rtt)         # modified based on playing speed, adjusted, * speed
                freezing_fraction += np.maximum(rtt - self.buffer/playing_speed, 0.0)   # modified based on playing speed, real time, /speed
                self.buffer = np.maximum(0.0, self.buffer - playing_speed*rtt)          # modified based on playing speed, adjusted, * speed
                # chech whether enter freezing
                if freezing_fraction > 0.0:
                    self.state = 2
            else:
                freezing_fraction += rtt    # in ms
        # Chunk downloading
        while True:
            throughput = self.throughput_trace[self.time_idx]   # in Mbps or Kbpms
            duration = self.time_trace[self.time_idx] * MS_IN_S - self.last_trace_time      # in ms
            deliverable_size = throughput * duration * PACKET_PAYLOAD_PORTION   # in Kbits      
            # Will also check whether freezing time exceeds the TOL
            if deliverable_size + chunk_sent > chunk_size:
                fraction = (chunk_size - chunk_sent) / (throughput * PACKET_PAYLOAD_PORTION)    # in ms, real time
                if self.state == 1:
                    assert freezing_fraction == 0.0
                    temp_freezing = np.maximum(fraction - self.buffer/playing_speed, 0.0)       # modified based on playing speed
                    if temp_freezing > self.latency_tol:
                        # should not happen
                        time_out = 1
                        self.last_trace_time += self.buffer/playing_speed + self.freezing_tol
                        downloading_fraction += self.buffer/playing_speed + self.freezing_tol
                        self.playing_time += self.buffer
                        chunk_sent += (self.freezing_tol + self.buffer/playing_speed) * throughput * PACKET_PAYLOAD_PORTION # in Kbits  
                        self.state = 0
                        self.buffer = 0.0
                        assert chunk_sent < chunk_size
                        return chunk_sent, downloading_fraction, freezing_fraction, time_out, start_state

                    downloading_fraction += fraction
                    self.last_trace_time += fraction
                    freezing_fraction += np.maximum(fraction - self.buffer/playing_speed, 0.0)  # modified based on playing speed 
                    self.playing_time += np.minimum(self.buffer, playing_speed*fraction)        # modified based on playing speed 
                    self.buffer = np.maximum(self.buffer - playing_speed*fraction, 0.0)         # modified based on playing speed 
                    if np.round(self.playing_time + self.buffer, 2) == np.round(chunk_start_time, 2):
                        self.buffer += self.chunk_duration * num_chunk
                    else:
                        # Should not happen in normal case, this is constrain for training
                        self.buffer = self.chunk_duration * num_chunk
                        self.playing_time = chunk_start_time
                    break
                # Freezing
                elif self.state == 2:
                    assert self.buffer == 0.0
                    if freezing_fraction + fraction > self.freezing_tol:
                        time_out = 1
                        self.last_trace_time += self.freezing_tol - freezing_fraction
                        downloading_fraction += self.freezing_tol - freezing_fraction
                        chunk_sent += (self.freezing_tol - freezing_fraction) * throughput * PACKET_PAYLOAD_PORTION # in Kbits
                        freezing_fraction = self.freezing_tol
                        self.state = 0
                        assert chunk_sent < chunk_size
                        return chunk_sent, downloading_fraction, freezing_fraction, time_out, start_state
                    freezing_fraction += fraction
                    self.last_trace_time += fraction
                    downloading_fraction += fraction
                    self.buffer += self.chunk_duration * num_chunk
                    self.playing_time = chunk_start_time
                    self.state = 1
                    break

                else:
                    assert self.buffer < self.start_up_th
                    # if freezing_fraction + fraction > self.freezing_tol:
                    #   self.buffer = 0.0
                    #   time_out = 1
                    #   self.last_trace_time += self.freezing_tol - freezing_fraction   # in ms
                    #   downloading_fraction += self.freezing_tol - freezing_fraction
                    #   chunk_sent += (self.freezing_tol - freezing_fraction) * throughput * PACKET_PAYLOAD_PORTION # in Kbits
                    #   freezing_fraction = self.freezing_tol
                    #   # Download is not finished, chunk_size is not the entire chunk
                    #   # print()
                    #   assert chunk_sent < chunk_size
                    #   return chunk_sent, downloading_fraction, freezing_fraction, time_out, start_state
                    downloading_fraction += fraction
                    self.buffer += self.chunk_duration * num_chunk
                    freezing_fraction += fraction
                    self.last_trace_time += fraction
                    if self.buffer >= self.start_up_th:
                        # Because it might happen after one long freezing (not exceed freezing tol)
                        # And resync, enter initial phase
                        buffer_end_time = chunk_start_time + self.chunk_duration * num_chunk
                        self.playing_time = buffer_end_time - self.buffer
                        # print buffer_end_time, self.buffer, " This is playing time"
                        self.state = 1
                    break

            # One chunk downloading does not finish
            # traceing
            if self.state == 1:
                assert freezing_fraction == 0.0
                temp_freezing = np.maximum(duration - self.buffer/playing_speed, 0.0)       # modified based on playing speed
                self.playing_time += np.minimum(self.buffer, playing_speed*duration)        # modified based on playing speed
                # Freezing time exceeds tolerence
                if temp_freezing > self.freezing_tol:
                    # should not happen
                    time_out = 1
                    self.last_trace_time += self.freezing_tol + self.buffer/playing_speed
                    downloading_fraction += self.freezing_tol + self.buffer/playing_speed
                    freezing_fraction = self.freezing_tol
                    self.playing_time += self.buffer
                    self.buffer = 0.0
                    # exceed TOL, enter startup, freezing time equals TOL
                    self.state = 0
                    chunk_sent += (self.freezing_tol + self.buffer/playing_speed) * throughput * PACKET_PAYLOAD_PORTION # in Kbits
                    assert chunk_sent < chunk_size
                    return chunk_sent, downloading_fraction, freezing_fraction, time_out, start_state

                chunk_sent += duration * throughput * PACKET_PAYLOAD_PORTION    # in Kbits
                downloading_fraction += duration    # in ms
                self.last_trace_time = self.time_trace[self.time_idx] * MS_IN_S # in ms
                self.time_idx += 1
                if self.time_idx >= len(self.time_trace):
                    self.time_idx = 1
                    self.last_trace_time = 0.0  # in ms
                self.buffer = np.maximum(self.buffer - playing_speed*duration, 0.0)         # modified based on playing speed
                # update buffer and state
                if temp_freezing > 0:
                    # enter freezing
                    self.state = 2
                    assert self.buffer == 0.0
                    freezing_fraction += temp_freezing

            # Freezing during trace
            elif self.state == 2:
                assert self.buffer == 0.0
                if duration + freezing_fraction > self.freezing_tol:
                    time_out = 1
                    self.last_trace_time += self.freezing_tol - freezing_fraction   # in ms
                    self.state = 0
                    downloading_fraction += self.freezing_tol - freezing_fraction
                    chunk_sent += (self.freezing_tol - freezing_fraction) * throughput * PACKET_PAYLOAD_PORTION # in Kbits
                    freezing_fraction = self.freezing_tol
                    # Download is not finished, chunk_size is not the entire chunk
                    assert chunk_sent < chunk_size
                    return chunk_sent, downloading_fraction, freezing_fraction, time_out, start_state

                freezing_fraction += duration   # in ms
                chunk_sent += duration * throughput * PACKET_PAYLOAD_PORTION    # in kbits
                downloading_fraction += duration    # in ms
                self.last_trace_time = self.time_trace[self.time_idx] * MS_IN_S # in ms
                self.time_idx += 1
                if self.time_idx >= len(self.time_trace):
                    self.time_idx = 1
                    self.last_trace_time = 0.0  # in ms
            # Startup
            else:
                assert self.buffer < self.start_up_th
                # if freezing_fraction + duration > self.freezing_tol:
                #   self.buffer = 0.0
                #   time_out = 1
                #   self.last_trace_time += self.freezing_tol - freezing_fraction   # in ms
                #   downloading_fraction += self.freezing_tol - freezing_fraction
                #   chunk_sent += (self.freezing_tol - freezing_fraction) * throughput * PACKET_PAYLOAD_PORTION # in Kbits
                #   freezing_fraction = self.freezing_tol
                #   # Download is not finished, chunk_size is not the entire chunk
                #   assert chunk_sent < chunk_size
                #   return chunk_sent, downloading_fraction, freezing_fraction, time_out, start_state
                chunk_sent += duration * throughput * PACKET_PAYLOAD_PORTION
                downloading_fraction += duration
                self.last_trace_time = self.time_trace[self.time_idx] * MS_IN_S # in ms
                self.time_idx += 1
                if self.time_idx >= len(self.time_trace):
                    self.time_idx = 1
                    self.last_trace_time = 0.0  # in ms
                freezing_fraction += duration
        # Finish downloading
        # if self.buffer > BUFFER_TH:
        #   # Buffer is too long, need sleep
        #   sleep = np.ceil((self.buffer - BUFFER_TH)/SLEEP_STEP) * SLEEP_STEP
        #   self.buffer -= sleep
        #   temp_sleep = sleep
        #   while True:
        #       duration = self.time_trace[self.time_idx] * MS_IN_S - self.last_trace_time
        #       if duration > temp_sleep:
        #           self.last_trace_time += temp_sleep
        #           break
        #       temp_sleep -= duration
        #       self.last_trace_time = self.time_trace[self.time_idx]
        #       self.time_idx += 1
        #       if self.time_idx >= len(self.time_trace):
        #           self.time_idx = 1
        #           self.last_trace_time = 0.0
        #   assert self.state == 1
        return chunk_size, downloading_fraction, freezing_fraction, time_out, start_state

