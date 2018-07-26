import theano.tensor as tt

def durw_model(inp, v, alpha, kappa, l_subs):
    """ Theano implementation of DU-RW model.

        inp -> input variable containing choices, 
               outcomes, and a control value
        v -> choice values from the previous time step
        alpha -> learning rate
        kappa -> coupling stength
        l_subs -> list of subject numbers
    """

    r = inp[:,0] # response/choice in current trial for all subjects
    o = 2*inp[:,1]-1 # reward outcome of current choice for all subjects
    t = inp[:,2] # a control variable which prevents update in case of nan trials

    v = tt.set_subtensor(v[l_subs, r], v[l_subs, r] \
        + t*alpha*(o-v[l_subs, r]))
    v = tt.set_subtensor(v[l_subs, 1-r], v[l_subs, 1-r] \
        + t*kappa*alpha*(-o-v[l_subs, 1-r]))

    return v



def edhmm_model(inp, pars, joint, pd0, P, l_subs):
	""" Theano implementation of edhmm model.

		inp -> input variable containing choices, 
		   outcomes, and a control value
		pars -> tensor containing parameters a^A_t, a^B_t, b^A_t, b^B_t
			which defined beliefs over reward probability of different choices
		joint -> joint probability distribution over states and durations \tilde{p}(s_t, d_t)
		pd0 -> prior beliefs over state durations p_0(d)
		P -> permutation matrix
        l_subs -> list of subject numbers
	"""
	r = inp[:, 0] # response/choice in current trial for all subjects
	o = inp[:, 1] # reward outcome of current choice for all subjects
	obs = tt.stack([o, 1-o])
	t = inp[:, 2] # a control variable which prevents update in case of nan trials

	marg = joint.sum(axis = -1) #marginal probability \tilde{p}(s_t) 
	cond = joint.T/marg.T # conditional probability \tilde{p}(d_t|s_t)
	theta = marg[:,0] # probability of being in a non reversal state

	nus = pars.sum(axis = -1) #scale parameter of the beta distribution
	mus = pars/nus[:,:,None] #mean of the beta distribution
	ollNR = (tt.log(mus[l_subs,r]).T*obs).sum(axis = 0) #observation log likeilhood in a no reversal state
	ollR = (tt.log(mus[l_subs, 1-r]).T*obs).sum(axis = 0) #observation log likelihood in reversal state

	theta = t*theta/(theta + tt.exp(ollR-ollNR)*(1-theta)) + (1-t)*theta

	pars = tt.set_subtensor(pars[l_subs, r], \
	                            pars[l_subs, r] + (t*obs*theta).T)
	pars = tt.set_subtensor(pars[l_subs, 1-r], \
	                            pars[l_subs, 1-r] + (t*obs*(1-theta)).T)

	#compute expected joint probability at the next time step \tilde{p}(s_{t+1}, d_{t+1})
	post = (cond*tt.stack([theta, 1-theta])).T
	post = tt.set_subtensor(post[:,:,0], post[:,:,0].dot(P))
	joint = post[:,:,0, None]*pd0[:,None,:]
	joint = tt.set_subtensor(joint[:,:,:-1], joint[:,:,:-1] + post[:,:,1:])

	return pars, joint