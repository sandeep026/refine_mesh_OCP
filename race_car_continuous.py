import casadi as cs
import matplotlib.pyplot as plt
import numpy as np


def racecar_point_wise():
    N = 25
    opti = cs.Opti()
    nx = 2
    nu = 1
    # ---- decision variables ----
    X = opti.variable(nx, N + 1)
    U = opti.variable(nu, N)
    t0 = 0
    tf = opti.variable(1, 1)
    cgl = cs.hcat([-cs.cos(cs.pi * i / N) for i in range(N + 1)])
    taucgl = 0.5 * (1 + cgl)
    tauuni = cs.linspace(0, 1, N + 1).T
    t = t0 + (tf - t0) * tauuni
    dt = cs.diff(t)

    # Optimize the duration of each interval directly
    xs = cs.SX.sym("xs", nx, 1)
    us = cs.SX.sym("us", nu, 1)
    ts = cs.SX.sym("ts")
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    dae = {
        "x": xs,
        "t": ts,
        "p": cs.vertcat(us, t0s, tfs),
        "ode": (tfs - t0s) * cs.vertcat(xs[1], us - xs[1]),
    }
    cvodes = cs.integrator(
        "integrator",
        "collocation",
        dae,
        0,
        1,
        {"simplify": True, "number_of_finite_elements": 1},
    )
    ires = cvodes.map(N, "serial")(x0=X[:, 0:-1], p=cs.vertcat(U, t[:, 0:-1], t[:, 1:]))

    defect = X[:, 1:] - ires["xf"]

    opti.minimize(tf)
    # x(t0)=0
    opti.subject_to(X[:, 0] == cs.DM([0] * 2))
    # pos
    opti.subject_to(X[0, -1] == 1)
    opti.subject_to(opti.bounded(0, tf, 10))
    opti.subject_to(defect == 0)
    opti.subject_to(opti.bounded(0, U, 1))
    opti.subject_to(
        opti.bounded(
            -cs.DM.inf(), X[1, :] - (1 - 0.5 * (cs.sin(2 * cs.pi * X[0, :]))), 0
        )
    )

    # ---- initial guesses ----
    Xg = cs.vertcat(cs.linspace(0, 1, N + 1).T, cs.linspace(0, 1, N + 1).T)
    Ug = cs.linspace(0.5, 0.5, N).T
    tg = cs.linspace(0, 1, N + 1).T
    opti.set_initial(tf, 5)
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)

    # ---- solve ----
    opti.solver(
        "ipopt",
    )

    sol = opti.solve()
    # ---- post-processing        ------
    from pylab import plot, step, figure, legend, show, spy

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

    # Evaluate violation at each interpolated state
    vio = cs.Function(
        "vio",
        [xs],
        [cs.mmax(cs.hcat([0, xs[1] - (1 - 0.5 * (cs.sin(2 * cs.pi * xs[0])))])) ** 2],
    )
    vio_vals = np.array([float(vio(Xip[:, k])) for k in range(tip.size)])

    plot(tip, vio_vals, "o-")

    show()


def racecar_ct_h():
    # integral constraint on path constriant alone (control bounds not included)
    # sparse grid intentionally used
    N = 25
    opti = cs.Opti()
    nx = 2 + 1
    nu = 1
    # ---- decision variables ----
    X = opti.variable(nx, N + 1)
    U = opti.variable(nu, N)
    t0 = 0
    tf = opti.variable(1, 1)
    cgl = cs.hcat([-cs.cos(cs.pi * i / N) for i in range(N + 1)])
    taucgl = 0.5 * (1 + cgl)
    tauuni = cs.linspace(0, 1, N + 1).T
    t = t0 + (tf - t0) * tauuni
    dt = cs.diff(t)

    # Optimize the duration of each interval directly
    xs = cs.SX.sym("xs", nx, 1)
    us = cs.SX.sym("us", nu, 1)
    ts = cs.SX.sym("ts")
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    dae = {
        "x": xs,
        "t": ts,
        "p": cs.vertcat(us, t0s, tfs),
        "ode": (tfs - t0s)
        * cs.vertcat(
            xs[1],
            us - xs[1],
            cs.fmax(0, xs[1] - (1 - 0.5 * (cs.sin(2 * cs.pi * xs[0])))) ** 2,
        ),
    }
    cvodes = cs.integrator(
        "integrator",
        "collocation",
        dae,
        0,
        1,
        {"simplify": True, "number_of_finite_elements": 1},
    )
    ires = cvodes.map(N, "serial")(x0=X[:, 0:-1], p=cs.vertcat(U, t[:, 0:-1], t[:, 1:]))

    defect = X[:, 1:] - ires["xf"]

    opti.minimize(tf)
    # x(t0)=0 + ct
    opti.subject_to(X[:, 0] == cs.DM([0] * 3))
    # pos
    opti.subject_to(X[0, -1] == 1)
    # ct
    opti.subject_to(X[2, -1] == 0)

    opti.subject_to(opti.bounded(0, tf, 10))
    opti.subject_to(defect == 0)
    opti.subject_to(opti.bounded(0, U, 1))

    # ---- initial guesses ----
    Xg = cs.vertcat(
        cs.linspace(0, 1, N + 1).T,
        cs.linspace(0, 1, N + 1).T,
        cs.linspace(0, 0, N + 1).T,
    )
    Ug = cs.linspace(0.5, 0.5, N).T
    tg = cs.linspace(0, 1, N + 1).T
    opti.set_initial(tf, 5)
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)

    # ---- solve ----
    opti.solver(
        "ipopt",
    )

    sol = opti.solve()
    # ---- post-processing        ------
    from pylab import plot, step, figure, legend, show, spy

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

    # Evaluate violation at each interpolated state
    vio = cs.Function(
        "vio",
        [xs],
        [cs.mmax(cs.hcat([0, xs[1] - (1 - 0.5 * (cs.sin(2 * cs.pi * xs[0])))])) ** 2],
    )
    vio_vals = np.array([float(vio(Xip[:, k])) for k in range(tip.size)])

    plot(tip, vio_vals, "o-")

    show()


def racecar_ct_h_u():
    # integral constraint on path constriant and control bounds
    # sparse grid intentionally used
    N = 24
    opti = cs.Opti()
    nx = 2 + 1 + 1 + 1
    nu = 1
    # ---- decision variables ----
    X = opti.variable(nx, N + 1)
    U = opti.variable(nu, N)
    t0 = 0
    tf = opti.variable(1, 1)
    cgl = cs.hcat([-cs.cos(cs.pi * i / N) for i in range(N + 1)])
    taucgl = 0.5 * (1 + cgl)
    tauuni = cs.linspace(0, 1, N + 1).T
    t = t0 + (tf - t0) * tauuni

    # Optimize the duration of each interval directly
    xs = cs.SX.sym("xs", nx, 1)
    us = cs.SX.sym("us", nu, 1)
    ts = cs.SX.sym("ts")
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    dae = {
        "x": xs,
        "t": ts,
        "p": cs.vertcat(us, t0s, tfs),
        "ode": (tfs - t0s)
        * cs.vertcat(
            xs[1],
            us - xs[1],
            cs.fmax(0, xs[1] - (1 - 0.5 * (cs.sin(2 * cs.pi * xs[0])))) ** 2,
            cs.fmax(0, us - 1) ** 2,
            cs.fmax(0, 0 - us) ** 2,
        ),
    }
    cvodes = cs.integrator(
        "integrator",
        "rk",
        dae,
        0,
        1,
        {"simplify": True, "number_of_finite_elements": 1},
    )
    ires = cvodes.map(N, "serial")(x0=X[:, 0:-1], p=cs.vertcat(U, t[:, 0:-1], t[:, 1:]))

    defect = X[:, 1:] - ires["xf"]

    opti.minimize(tf)
    # x(t0)=0 + ct
    opti.subject_to(X[:, 0] == cs.DM([0] * 5))
    # pos
    opti.subject_to(X[0, -1] == 1)
    # ct
    opti.subject_to(X[2:5, -1] == X[2:5, 0])

    opti.subject_to(opti.bounded(0.5, tf, 10))
    opti.subject_to(defect == 0)

    # ---- initial guesses ----
    Xg = cs.vertcat(
        cs.linspace(0, 1, N + 1).T,
        cs.linspace(0, 1, N + 1).T,
        cs.linspace(0, 0, N + 1).T,
        cs.linspace(0, 0, N + 1).T,
        cs.linspace(0, 0, N + 1).T,
    )
    Ug = cs.linspace(0.5, 0.5, N).T
    opti.set_initial(tf, 5)
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)

    # ---- solve ----
    opti.solver(
        "ipopt",
    )

    sol = opti.solve()
    # ---- post-processing        ------
    from pylab import plot, step, figure, legend, show, spy

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

    # Evaluate violation at each interpolated state
    vio = cs.Function(
        "vio",
        [xs],
        [cs.fmax(0, xs[1] - (1 - 0.5 * (cs.sin(2 * cs.pi * xs[0])))) ** 2],
    )
    vio_vals = np.array([float(vio(Xip[:, k])) for k in range(tip.size)])

    plot(tip, vio_vals, "o-")

    show()

racecar_point_wise()
racecar_ct_h()
racecar_ct_h_u()
