import casadi as cs
import numpy as np

def hypersensitive_dilation():
    N = 1000
    opti = cs.Opti()
    nx=1+1
    nu=1+1
    X = opti.variable(nx, N + 1)
    U = opti.variable(nu, N)
    # Optimize the duration of each interval directly
    t0=0
    tf=10000
    tau0=0
    tauf=1
    tau=cs.linspace(tau0,tauf,N+1).T

    xs=cs.SX.sym('xs',nx,1)
    us=cs.SX.sym('us',nu,1)
    ts=cs.SX.sym('ts')
    t0s=cs.SX.sym('t0s')
    tfs=cs.SX.sym('tfs')
    dae = {'x': xs, 't':ts,'p': cs.vertcat(us,t0s,tfs), 'ode': (tfs-t0s)*cs.vertcat((-xs[0]**3+us[0])*us[1],us[1]),'quad':0.5*(tfs-t0s)*(xs[0]**2+us[0]**2)*us[1]}
    cvodes = cs.integrator('integrator', 'cvodes', dae,{'max_num_steps': 20000})
    #cvodes=cs.integrator('integrator', 'collocation', dae,{'interpolation_order':1})
    ires=cvodes.map(N,'serial')(x0=X[:,0:-1],p=cs.vertcat(U,tau[:,0:-1],tau[:,1:]))
    opti.subject_to(X[:,1:]-ires['xf']==0,cs.repmat(cs.DM([1,tf]),1,N))
    opti.minimize(cs.sum2(ires['qf']))    
    opti.subject_to(X[:,0] == cs.vertcat(1.5,t0),cs.DM([1,tf]) )
    opti.subject_to(X[:,-1] == cs.vertcat(1,tf),cs.DM([1,tf]))
    eps=1e-16
    opti.subject_to(opti.bounded(0+eps,U[1,:],cs.DM.inf()))

    c1=0.1
    Xg=cs.vertcat(cs.linspace(c1,c1,N+1).T,cs.linspace(t0,tf,N+1).T)
    Ug=cs.vertcat(cs.linspace(c1,c1,N).T**3,cs.linspace(tf,tf,N).T)
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)

    # ---- solve ----
    opti.solver(
        "ipopt",
        {
            "ipopt.max_iter": 3000,
            "ipopt.tol": 1e-4,
            "ipopt.hessian_approximation": 'exact'
        }
    )

    sol = opti.solve()
    t = np.array(sol.value(tau)).flatten()
    X = np.array(sol.value(X))
    U = np.array(sol.value(U))
    
    # ---- plots ----
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(t, X[0,:], "r-^")
    plt.xlabel('$\tau$')
    plt.ylabel('x')
    plt.figure()
    plt.step(t[:-1], U[0,:], "b-o")
    plt.xlabel('$\tau$')
    plt.ylabel('u')
    plt.figure()
    plt.plot(t, X[1,:], "g-s")
    plt.xlabel('$\tau$')
    plt.ylabel('t')
    plt.figure()
    plt.step(t[:-1], U[1,:], "b-")
    plt.xlabel('$\tau$')
    plt.ylabel('s (dilation)')
    plt.show()

hypersensitive_dilation()