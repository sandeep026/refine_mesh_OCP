import casadi as cs
import matplotlib.pyplot as plt
import numpy as np







def racecar(N=25,path:bool=False,con:bool=False,mesh:str|None='uni'):
    # integral constraint on path constriant and control bounds
    # sparse grid intentionally used
    opti = cs.Opti()
    nx,nu=get_dims(path,con)
    X = opti.variable(nx, N + 1)
    U = opti.variable(nu, N)
    t0 = 0
    tf = opti.variable(1, 1)
    tau= get_tau(N,mesh)
    t = t0 + (tf - t0) * tau
    xr, vio=get_integrator(path,con,nx,nu)

    ires = xr.map(N, "serial")(x0=X[:, 0:-1], p=cs.vertcat(U, t[:, 0:-1], t[:, 1:]))

    defect = X[:, 1:] - ires["xf"]

    opti.minimize(tf)
    # pos(t0)=0 vel(t0)=0 
    # pos (tf)=1
    opti.subject_to(X[0:2, 0] == cs.DM([0] * 2))
    opti.subject_to(X[0, -1] == 1)
    opti.subject_to(opti.bounded(0.5, tf, 10))
    opti.subject_to(defect == 0)
    if con is False:
        opti.subject_to(opti.bounded(0,U,1))
    
    if path is False:
        opti.subject_to(opti.bounded(-cs.DM.inf(),X[1, :] - (1 - 0.5 * (cs.sin(2 * cs.pi * X[0, :]))),0))

    # ct
    if path is True or con is True:
        opti.subject_to(X[2:, 0] == cs.DM.zeros(X[2:, 0].shape))
        opti.subject_to(X[2:, -1] == X[2:, 0])
    
    Xg,Ug=get_guess(N,path,con)
    opti.set_initial(tf, 5)
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)

    # ---- solve ----
    opti.solver(
        "ipopt",
        {'ipopt.max_iter':3000}
    )

    sol = opti.solve()
    plot_soln(sol,X,U,t,vio)

def get_dims(path,con):
    nxh=1 if path else 0
    nxu=2 if con else 0
    nx = 2 + nxh+nxu
    nu = 1
    return nx,nu

def get_tau(N,mesh):
    if mesh=='uni':
        tau = cs.linspace(0, 1, N + 1).T
    elif mesh=='cgl':
        cgl = cs.hcat([-cs.cos(cs.pi * i / N) for i in range(N + 1)])
        tau = 0.5 * (1 + cgl)
    else: 
        raise ValueError('uni or cgl are valid')
    return tau

def get_integrator(path,con,nx,nu):
    # Optimize the duration of each interval directly
    xs = cs.SX.sym("xs", nx, 1)
    us = cs.SX.sym("us", nu, 1)
    ts = cs.SX.sym("ts")
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    def fnew(f):
        eps=1e-16
        return (0.5*(f+cs.sqrt(f**2+eps**2)))**2
        #return (cs.fmax(0,f))

    aug_ode=[]
    if path:
        aug_ode.append(fnew(xs[1] - (1 - 0.5 * (cs.sin(2 * cs.pi * xs[0])))))
    if con:
        aug_ode.append(fnew(us - 1))
        aug_ode.append(fnew(0-us))

#    if path:
#        aug_ode.append(cs.fmax(0, xs[1] - (1 - 0.5 * (cs.sin(2 * cs.pi * xs[0])))) ** 2)
#    if con:
#        aug_ode.append(cs.fmax(0, us - 1) ** 2)
#        aug_ode.append(cs.fmax(0, 0-us) ** 2)
    ode=[xs[1],us - xs[1]]
    dae = {
        "x": xs,
        "t": ts,
        "p": cs.vertcat(us, t0s, tfs),
        "ode": (tfs - t0s)*cs.vcat(ode+aug_ode)
    }

    xr = cs.integrator(
        "integrator",
        "rk",
        dae,
        0,
        1,
        {"simplify": True, "number_of_finite_elements": 1},
    )

    # Evaluate violation at each interpolated state
    vio = cs.Function(
        "vio",
        [xs],
        [cs.fmax(0, xs[1] - (1 - 0.5 * (cs.sin(2 * cs.pi * xs[0])))) ** 2],
    )

    return xr, vio

def get_guess(N:int,path,con):
    if path is False and con is False:
        X_aug_g=[]
    else:
        c1=1 if path else 0
        c2=2 if con else 0
        X_aug_g=[cs.linspace(0, 0, N + 1).T]*(c1+c2)

    Xg = cs.vcat([cs.linspace(0, 1, N + 1).T]*2+X_aug_g)
    Ug = cs.linspace(0.5, 0.5, N).T
    return Xg,Ug

def plot_soln(sol:cs.OptiSol,X:cs.MX,U:cs.MX,t:cs.MX,vio:cs.Function):
    from pylab import figure, legend, plot, show, step

    tnlp = sol.value(t)
    posnlp = sol.value(X[0, :])
    velnlp = sol.value(X[1, :])
    pathnlp = sol.value(1 - 0.5 * cs.sin(2 * cs.pi * X[0, :]))

    plot(tnlp, posnlp, "o-", label="pos")
    plot(tnlp, velnlp, "o-", label="speed")
    plot(
        tnlp,
        pathnlp,
        "r--",
        label="speed limit",
    )
    step(tnlp[0:-1], sol.value(U), "k", label="throttle")
    legend(loc="upper left")

    figure()
    td = np.asarray(tnlp).ravel()
    Xd = np.asarray(sol.value(X))  # shape: (nx, N+1)

    tip = np.linspace(td[0], td[-1], 200)

    # Piecewise-linear interpolation of each state
    Xip = np.vstack([np.interp(tip, td, Xd[i, :]) for i in range(Xd.shape[0])])


    vio_vals = np.array([float(vio(Xip[:, k])) for k in range(tip.size)])

    plot(tip, vio_vals, "o-")

    show()


#racecar_point_wise()
#racecar_ct_h_u()
racecar(N=300,path=True,con=True,mesh='uni')
