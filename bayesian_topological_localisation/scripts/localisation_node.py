#!/usr/bin/env python
import rospy
import threading
import numpy as np
import matplotlib.pyplot as plt

from bayesian_topological_localisation.particle_filter import TopologicalParticleFilter
from bayesian_topological_localisation.prediction_model import PredictionModel
from bayesian_topological_localisation.srv import LocaliseAgent, LocaliseAgentRequest, \
    LocaliseAgentResponse, StopLocalise, StopLocaliseRequest, StopLocaliseResponse, \
    UpdatePoseObservation, UpdatePoseObservationRequest, UpdatePoseObservationResponse, \
    UpdateLikelihoodObservation, UpdateLikelihoodObservationRequest, UpdateLikelihoodObservationResponse, \
    UpdatePriorLikelihoodObservation, UpdatePriorLikelihoodObservationRequest, UpdatePriorLikelihoodObservationResponse, \
    Predict, PredictRequest, PredictResponse, SetFloat64, SetFloat64Request, SetFloat64Response
from bayesian_topological_localisation.msg import DistributionStamped, PoseObservation, LikelihoodObservation, ParticlesState
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from strands_navigation_msgs.msg import TopologicalMap
from std_msgs.msg import String

class TopologicalLocalisation():

    def __init__(self):
        # agents currently tracking
        self.agents = []
        # observation subscribers for each agent
        self.obs_subscribers = []
        # # publishers localisation result for each agent
        self.res_publishers = []
        # # publishers viz result for each agent
        self.viz_publishers = []
        # services for updating the state estimation
        self.upd_services = []
        # thread that loop predictions at fixed rate for each agent
        # self.prediction_threads = []
        # # contains the particle filters for each agent
        # self.pfs = []
        # agents colors
        self.agents_colors = []

        # these will contain info about the topology
        self.topo_map = None
        self.node_diffs2D = []
        self.node_distances = []
        self.connected_nodes = []
        self.node_names = []
        self.node_coords = []

        # contains a list of threading.Event for stopping the localisation of each agent
        self.stopping_events = []

        # to avoid inconsistencies when registering/unregistering agents concurrently
        self.internal_lock = threading.Lock()

        # default values for pf
        self.default_reinit_jsd_threshold = 0.975
        self.default_unconnected_jump_threshold = 0.4

        ## set default values ## TODO all this should be passed from launch
        # default name is unknown if requested is ''
        # name = (request.name, 'unknown')[request.name == '']
        # default particles number is 300 if requested is 0
        self.n_particles = 5000 # (request.n_particles, 3000)[self.n_particles <= 0]
        self.initial_spread_policy = 0
        self.prediction_model = 0
        self.do_prediction = True
        # default prediction rate is 0.5 if requested is 0.
        self.prediction_rate =  0.5 #(request.prediction_rate, 0.5)[request.prediction_rate <= 0.]
        # default speed decay is 1 if requested is 0
        self.prediction_speed_decay = 1.0

        # declare services
        # rospy.Service("~localise_agent", LocaliseAgent, self._localise_agent_handler)
        rospy.Service("~stop_localise", StopLocalise, self._stop_localise_handler)
        rospy.Service("~set_JSD_upper_bound", SetFloat64, self._set_JSD_upper_bound)
        rospy.Service("~set_entropy_lower_bound", SetFloat64, self._set_entropy_lower_bound)

        rospy.Subscriber("topological_map", TopologicalMap, self._topo_map_cb)

        rospy.loginfo("Waiting for topological map...")
        while self.topo_map is None:
            rospy.sleep(0.5)

        rospy.loginfo("DONE")


    def _set_JSD_upper_bound(self, request):
        self.default_reinit_jsd_threshold = request.value
        for pf in self.pfs:
            self.pf.set_JSD_upper_bound(request.value)

        return SetFloat64Response(True)

    def _set_entropy_lower_bound(self, request):
        self.default_unconnected_jump_threshold = request.value
        for pf in self.pfs:
            self.pf.set_entropy_lower_bound(request.value)

        return SetFloat64Response(True)

    def _marker_factory(self, agent_name, particle=True):
        marker = Marker()
        if particle:
            # the particle
            marker.header.frame_id = "/map"
            marker.type = marker.SPHERE
            marker.pose.position.z = 0
            marker.pose.orientation.w = 1
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = self.agents_colors[self.agents.index(agent_name)][1][0]
            marker.color.g = self.agents_colors[self.agents.index(agent_name)][1][1]
            marker.color.b = self.agents_colors[self.agents.index(agent_name)][1][2]
            marker.color.a = self.agents_colors[self.agents.index(agent_name)][1][3]
        else:
            marker.header.frame_id = "/map"
            marker.type = marker.SPHERE
            marker.pose.position.z = 1.2
            marker.pose.orientation.w = 1
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.r = self.agents_colors[self.agents.index(agent_name)][0][0]
            marker.color.g = self.agents_colors[self.agents.index(agent_name)][0][1]
            marker.color.b = self.agents_colors[self.agents.index(agent_name)][0][2]
            marker.color.a = self.agents_colors[self.agents.index(agent_name)][0][3]
        return marker

    def _initialise_new_agent(self, agent_name, color=None):
        rospy.loginfo("Initialising new agent {}".format(agent_name))
        # choose a color for the new agent
        _colors = [
            plt.cm.tab20((len(self.agents) * 2) % len(plt.cm.tab20.colors)),    # marker color
            plt.cm.tab20((len(self.agents) * 2 + 1) % len(plt.cm.tab20.colors)) # particles color
        ]

        self.agents.append(agent_name)
        self.agents_colors.append(_colors)

        # Initialize publishers and messages
        cn_pub = rospy.Publisher("{}/estimated_node".format(agent_name), String, queue_size=10, latch=True)
        pd_pub = rospy.Publisher("{}/current_prob_dist".format(agent_name), DistributionStamped, queue_size=10, latch=True)
        ptcs_pub = rospy.Publisher("{}/particles_states".format(agent_name), ParticlesState, queue_size=10, latch=True)
        self.res_publishers.append((cn_pub, pd_pub, ptcs_pub))
        cnviz_pub = rospy.Publisher("{}/estimated_node_viz".format(agent_name), Marker, queue_size=10)
        parviz_pub = rospy.Publisher("{}/particles_viz".format(agent_name), MarkerArray, queue_size=10)
        # staparviz_pub = rospy.Publisher("{}/stateless_particles_viz".format(agent_name), MarkerArray, queue_size=10)
        self.viz_publishers.append((cnviz_pub, parviz_pub))



    def _loop(self):
        rospy.loginfo("Starting loop")

        # Initialize the prediction model
        # if prediction_model == LocaliseAgentRequest.PRED_CTMC:
        self.pm = PredictionModel(
            pred_type=PredictionModel.CTMC,
            node_coords=self.node_coords,
            node_diffs2D=self.node_diffs2D,
            node_distances=self.node_distances,
            connected_nodes=self.connected_nodes
        )

        # Initialize a new instance of particle_filter
        self.pf = TopologicalParticleFilter(
            num=self.n_particles,
            prediction_model=self.pm,
            initial_spread_policy=self.initial_spread_policy,
            prediction_speed_decay=self.prediction_speed_decay,
            node_coords=self.node_coords,
            node_distances=self.node_distances,
            connected_nodes=self.connected_nodes,
            node_diffs2D=self.node_diffs2D,
            node_names=self.node_names,
            reinit_jsd_threshold=self.default_reinit_jsd_threshold,
            unconnected_jump_threshold=self.default_unconnected_jump_threshold
        )

        # publishers for visualizing all at once
        self.cnviz_pub = rospy.Publisher("~estimated_nodes_viz", MarkerArray, queue_size=10)
        self.parviz_pub = rospy.Publisher("~particles_viz", MarkerArray, queue_size=10)

        # subscribe to services for update
        self.upd_services.append({
            rospy.Service("~update_pose_obs", UpdatePoseObservation, self.__update_pose_handler),
            rospy.Service("~update_likelihood_obs", UpdateLikelihoodObservation, self.__update_likelihood_handler),
            # rospy.Service("~predict_stateless", Predict, self.__do_stateless_prediction),
            # rospy.Service("~update_stateless", UpdatePriorLikelihoodObservation, self.__do_stateless_update)
        })

        # subscribe to topics receiving observation
        self.obs_subscribers.append((
            rospy.Subscriber("~pose_obs", PoseObservation, self.__pose_obs_cb),
            rospy.Subscriber("~likelihood_obs",
                                LikelihoodObservation, self.__likelihood_obs_cb)
        ))

        rospy.loginfo("n_particles:{}, initial_spread_policy:{}, prediction_model:{}, do_prediction:{}, prediction_rate:{}. prediction_speed_decay:{}".format(
            self.n_particles, self.initial_spread_policy, self.prediction_model, self.do_prediction, self.prediction_rate, self.prediction_speed_decay
        ))

        self._initialise_new_agent("UNK")

        thr = None
        if self.do_prediction:
            rate = rospy.Rate(self.prediction_rate)
            while not rospy.is_shutdown():
                self.internal_lock.acquire()
                p_estimate, particles = self.pf.predict(
                    timestamp_secs=rospy.get_rostime().to_sec()
                )
                if not (p_estimate is None or particles is None): 
                    self.__publish(p_estimate, particles)
                self.internal_lock.release()
                rate.sleep()
        else:
            rospy.spin()

        rospy.loginfo("Exiting.")

    def __prepare_pd_msg(self, particles, timestamp=None):
        # print(particles)
        pdmsg = DistributionStamped()
        # print(particles)
        _nodes = [p.node for p in particles]
        nodes, counts = np.unique(_nodes, return_counts=True)

        probs = np.zeros((self.node_names.shape[0]))
        probs[nodes] = counts.astype(float) / np.sum(counts)

        if timestamp is None:
            timestamp = rospy.get_rostime()
        pdmsg.header.stamp = timestamp
        pdmsg.nodes = self.node_names.tolist()
        pdmsg.values = np.copy(probs).tolist()

        return pdmsg

    def __prepare_cn_msg(self, node):
        strmsg = String()
        strmsg.data = self.node_names[node]

        return strmsg

    def __prepare_ptcs_msg(self, particles):
        ptcsmsg = ParticlesState()
        ptcsmsg.nodes = [self.node_names[p.node] for p in particles]
        ptcsmsg.vels_x = [p.vel[0] for p in particles]
        ptcsmsg.vels_y = [p.vel[1] for p in particles]
        ptcsmsg.times = [p.life for p in particles]

        return ptcsmsg

    # function to publish current node and particles distribution
    def __publish(self, node, particles):
        particles_array = MarkerArray()
        nodes_array = MarkerArray()
        time = rospy.get_rostime()
        pa_i = 0
        for agent_i in range(len(self.agents)):
            # publish localisation results
            self.res_publishers[agent_i][0].publish(self.__prepare_cn_msg(node[agent_i]))
            self.res_publishers[agent_i][1].publish(self.__prepare_pd_msg(particles[agent_i]))
            self.res_publishers[agent_i][2].publish(self.__prepare_ptcs_msg(particles[agent_i]))

            # collects particles
            for p in particles[agent_i]:
                # print(p)
                ptc_mkr = self._marker_factory(self.agents[agent_i], particle=True)
                ptc_mkr.header.stamp = time
                ptc_mkr.pose.position.x = self.node_coords[p.node][0] + \
                    (ptc_mkr.scale.x * np.random.randn())
                ptc_mkr.pose.position.y = self.node_coords[p.node][1] + \
                    (ptc_mkr.scale.y * np.random.randn())
                ptc_mkr.id = pa_i
                particles_array.markers.append(ptc_mkr)
                pa_i += 1

            # collect estimated nodes
            node_mkr = self._marker_factory(self.agents[agent_i], particle=False)
            node_mkr.header.stamp = time
            node_mkr.pose.position.x = self.node_coords[node[agent_i]][0]
            node_mkr.pose.position.y = self.node_coords[node[agent_i]][1]
            nodes_array.markers.append(node_mkr)

        # publish viz stuff
        self.cnviz_pub.publish(nodes_array)
        self.parviz_pub.publish(particles_array)

    ## topic callbacks ##
    # send pose observation to particle filter
    def __pose_obs_cb(self, msg):
        rospy.loginfo("Received pose observations for {}".format(msg.identity))
        self.internal_lock.acquire()
        # check if it's a newly seen agent
        if msg.identity not in self.agents:
            self._initialise_new_agent(msg.identity)

        if np.isfinite(msg.pose.pose.pose.position.x) and \
                np.isfinite(msg.pose.pose.pose.position.y) and \
                np.isfinite(msg.pose.pose.covariance[0]) and \
                np.isfinite(msg.pose.pose.covariance[7]):
            p_estimated, particles = self.pf.receive_pose_obs(
                msg.pose.pose.pose.position.x,
                msg.pose.pose.pose.position.y,
                msg.pose.pose.covariance[0], # variance of x
                msg.pose.pose.covariance[7], # variance of y
                # (rospy.get_rostime().to_sec(), msg.pose.header.stamp.to_sec())[
                    # msg.pose.header.stamp.to_sec() > 0]
                rospy.get_rostime().to_sec(),
                identity=msg.identity
            )
            self.__publish(p_estimated, particles)
        else:
            rospy.logwarn(
                "Received non-admissible pose observation <{}, {}, {}, {}>, discarded".format(msg.pose.pose.pose.position.x, msg.pose.pose.pose.position.y, msg.pose.pose.covariance[0], msg.pose.pose.covariance[7]))
        self.internal_lock.release()

    # send likelihood observation to particle filter
    def __likelihood_obs_cb(self, msg):
        rospy.loginfo("Received likelihood observation for {}".format(msg.identity))
        self.internal_lock.acquire()
        # check if it's a newly seen agent
        if msg.identity not in self.agents:
            self._initialise_new_agent(msg.identity)

        if len(msg.likelihood.nodes) == len(msg.likelihood.values):
            try:
                nodes = [np.where(self.node_names == nname)[0][0] for nname in msg.likelihood.nodes]
            except IndexError:
                rospy.logwarn(
                    "Received non-admissible node name {}, likelihood discarded".format(msg.likelihood.nodes))
            else:
                values = np.array(msg.likelihood.values)
                if np.isfinite(values).all() and (values >= 0.).all() and np.sum(values) > 0:
                    p_estimated, particles = self.pf.receive_likelihood_obs(
                        nodes, 
                        msg.likelihood.values,
                        # (rospy.get_rostime().to_sec(), msg.likelihood.header.stamp.to_sec())[
                            # msg.likelihood.header.stamp.to_sec() > 0]
                        rospy.get_rostime().to_sec(),
                        identity=msg.identity
                    )
                    self.__publish(p_estimated, particles)
                else:
                    rospy.logwarn(
                        "Received non-admissible likelihood observation {}, discarded".format(msg.likelihood.values))
        else:
            rospy.logwarn("Nodes array and values array sizes do not match {} != {}, discarding likelihood observation".format(
                len(msg.likelihood.nodes), len(msg.likelihood.values)))
        self.internal_lock.release()

    ## Services handlers ##
    # Get the pose observation and returns the localisation result
    def __update_pose_handler(self, request):
        rospy.loginfo("Received pose request for {}".format(request.identity))
        self.internal_lock.acquire()
        # check if it's a newly seen agent
        if request.identity not in self.agents:
            self._initialise_new_agent(request.identity)

        if np.isfinite(request.pose.pose.pose.position.x) and \
                np.isfinite(request.pose.pose.pose.position.y) and \
                np.isfinite(request.pose.pose.covariance[0]) and \
                np.isfinite(request.pose.pose.covariance[7]):
            p_estimated, particles = self.pf.receive_pose_obs(
                request.pose.pose.pose.position.x,
                request.pose.pose.pose.position.y,
                request.pose.pose.covariance[0],  # variance of x
                request.pose.pose.covariance[7],  # variance of y
                # (rospy.get_rostime().to_sec(), request.pose.header.stamp.to_sec())[
                #     request.pose.header.stamp.to_sec() > 0]
                rospy.get_rostime().to_sec(),
                identity=request.identity
            )
            self.__publish(p_estimated, particles)
            resp = UpdatePoseObservationResponse()
            resp.success = True
            resp.estimated_node = p_estimated#self.__prepare_cn_msg(p_estimated.node).data
            for agent_i in range(len(self.agents)):
                resp.current_prob_dist.append(self.__prepare_pd_msg(particles[agetn_i]))
            self.internal_lock.release()
            return(resp)
        else:
            rospy.logwarn(
                "Received non-admissible pose observation <{}, {}, {}, {}>, discarded".format(request.pose.pose.pose.position.x, request.pose.pose.pose.position.y, request.pose.pose.covariance[0], request.pose.pose.covariance[7]))
            self.internal_lock.release()
        
        # fallback negative response
        resp = UpdatePoseObservationResponse()
        resp.success = False
        return(resp)

    # get a likelihood observation and return localisation result
    def __update_likelihood_handler(self, request):
        rospy.loginfo("Received likelihood request for {}".format(request.identity))
        self.internal_lock.acquire()
        # check if it's a newly seen agent
        if request.identity not in self.agents:
            self._initialise_new_agent(request.identity)

        if len(request.likelihood.nodes) == len(request.likelihood.values):
            try:
                nodes = [np.where(self.node_names == nname)[0][0]
                            for nname in request.likelihood.nodes]
            except IndexError:
                rospy.logwarn(
                    "Received non-admissible node name {}, likelihood discarded".format(request.likelihood.nodes))
            else:
                values = np.array(request.likelihood.values)
                # rospy.loginfo("Received likelihood: {}".format(zip(nodes, values)))
                if np.isfinite(values).all() and (values >= 0.).all() and np.sum(values) > 0:
                    p_estimated, particles = self.pf.receive_likelihood_obs(
                        nodes,
                        request.likelihood.values,
                        # (rospy.get_rostime().to_sec(), request.likelihood.header.stamp.to_sec())[
                            # request.likelihood.header.stamp.to_sec() > 0]
                        rospy.get_rostime().to_sec(),
                        identity=request.identity
                    )
                    self.__publish(p_estimated, particles, agent=request.identity)
                    resp = UpdateLikelihoodObservationResponse()
                    resp.success = True
                    resp.estimated_node = p_estimated#self.__prepare_cn_msg(p_estimated.node).data
                    for agent_i in range(len(self.agents)):
                        resp.current_prob_dist.append(self.__prepare_pd_msg(particles[agent_i]))
                    self.internal_lock.release()
                    return(resp)
                else:
                    rospy.logwarn(
                        "Received non-admissible likelihood observation {}, discarded".format(request.likelihood.values))
                    self.internal_lock.release()

        else:
            rospy.logwarn("Nodes array and values array sizes do not match {} != {}, discarding likelihood observation".format(
                len(request.likelihood.nodes), len(request.likelihood.values)))

        # fallback negative response
        resp = UpdateLikelihoodObservationResponse()
        resp.success = False
        return(resp)

    def _stop_localise_handler(self, request):
        rospy.loginfo("Unregistering agent {} for localisation".format(request.name))
        self.internal_lock.acquire()
        # default name is unknown if requested is ''
        name = (request.name, 'unknown')[request.name == '']
        if name in self.agents:
            agent_idx = self.agents.index(name)
            # stop prediction loop
            if self.stopping_events[agent_idx] is not None:
                self.stopping_events[agent_idx].set()
            if self.prediction_threads[agent_idx] is not None:
                self.prediction_threads[agent_idx].join()

            # unregister topic subs
            for sub in self.obs_subscribers[agent_idx]:
                sub.unregister()
            # shutting down services
            for srv in self.upd_services[agent_idx]:
                srv.shutdown()
            # unregister topic pubs
            for pub in self.res_publishers[agent_idx]:
                pub.unregister()
            for pub in self.viz_publishers[agent_idx]:
                pub.unregister()
            

            # cleanup all the related variables
            del self.stopping_events[agent_idx]
            del self.prediction_threads[agent_idx]
            del self.obs_subscribers[agent_idx]
            del self.res_publishers[agent_idx]
            del self.viz_publishers[agent_idx]
            del self.upd_services[agent_idx]
            del self.agents[agent_idx]

            self.internal_lock.release()
            rospy.loginfo("DONE")
            return StopLocaliseResponse(True)
        else:
            rospy.logwarn("The agent {} is already not being localised.".format(name))
            self.internal_lock.release()
            return StopLocaliseResponse(False)

    def _topo_map_cb(self, msg):
        """This function receives the Topological Map"""
        self.topo_map = msg

        # save and compute topological map informations
        self.node_names = np.array([node.name for node in self.topo_map.nodes])
        self.node_coords = np.array(
            [[node.pose.position.x, node.pose.position.y] for node in self.topo_map.nodes])

        self.node_diffs2D = []
        self.node_distances = []
        self.connected_nodes = []
        for i, _ in enumerate(self.node_names):
            self.node_diffs2D.append(self.node_coords - self.node_coords[i])
            self.connected_nodes.append(np.array([np.where(self.node_names == edge.node)[
                                        0][0] for edge in self.topo_map.nodes[i].edges]))

        self.node_diffs2D = np.array(self.node_diffs2D)
        self.connected_nodes = np.array(self.connected_nodes)
        
        self.node_distances = np.sqrt(np.sum(self.node_diffs2D ** 2, axis=2))
        
        # print("self.node_diffs2D", self.node_diffs2D.shape)
        # print("self.node_distances", self.node_distances.shape)
        # print("self.connected_nodes", self.connected_nodes.shape)

    def close(self):
        # stop all the threads
        for thr, stop_event in zip(self.prediction_threads, self.stopping_events):
            stop_event.set()
            thr.join()


if __name__ == "__main__":
    rospy.init_node("bayesian_topological_localisation")

    localisation_node = TopologicalLocalisation()

    localisation_node._loop()

    localisation_node.close()
