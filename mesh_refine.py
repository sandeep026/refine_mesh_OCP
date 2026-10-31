import casadi as cs
from pytrajopt import (
    Ocp,
    casadi_integrator,
    check_feasibility,
    generate_NLP,
    plot_solution,
    disc_opt,
)
import numpy as np


def hyper(tau):
    ocp_data = {}
    ocp_data["nx"] = 1
    ocp_data["nu"] = 1
    ts = cs.SX.sym("ts", 1, 1)
    xs = cs.SX.sym("xs", ocp_data["nx"], 1)
    us = cs.SX.sym("us", ocp_data["nu"], 1)
    t0s = cs.SX.sym("t0s", 1, 1)
    tfs = cs.SX.sym("tfs", 1, 1)
    x0s = cs.SX.sym("x0s", ocp_data["nx"], 1)
    xfs = cs.SX.sym("xfs", ocp_data["nx"], 1)
    path = [ts, xs, us]
    event = [t0s, tfs, x0s, xfs]
    ocp_data["L"] = cs.Function("L", path, [0.5 * (xs**2 + us**2)])
    ocp_data["E"] = None
    ocp_data["e"] = cs.Function("e", event, [cs.vertcat(x0s - 1.5, xfs - 1)])
    ocp_data["e_lb"] = cs.DM([0] * 2)
    ocp_data["e_ub"] = cs.DM([0] * 2)
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = cs.DM(10000)
    ocp_data["f"] = cs.Function("f", path, [-(xs**3) + us])
    ocp = Ocp(**ocp_data)
    nlp = cs.Opti()
    # n=3200
    # cgl=cs.DM([-cs.cos(k*cs.pi/n) for k in range(n+1)]).T
    # cgl01=0.5*(1+cgl)
    n1 = 100
    n2 = 5
    n3 = 25
    n = n1 + n2 + n3 - 1
    eps = 1e-8
    rao = cs.hcat(
        [cs.linspace(-1, -0.996 - eps, n1).T]
        + [cs.linspace(-0.996 + eps, 0.996 - eps, n2).T]
        + [cs.linspace(0.996 + eps, 1, n3).T]
    )
    rao01 = 0.5 * (1 + rao)
    method = casadi_integrator(
        plugin="collocation",
        plugin_options={"number_of_finite_elements": 1, "simplify": True},
        nx=ocp.nx,
        nu=ocp.nu,
        path=[ts, xs, us],
        f=ocp.f(ts, xs, us),
    )
    disc = disc_opt(n=tau.size - 1, method=method, grid=cs.DM(tau).T)
    dis_ocp = generate_NLP(ocp, nlp, disc)
    nlp.solver("ipopt")
    Xg = cs.linspace(1.5, 1, tau.size).T
    Ug = cs.linspace(0.1, 0.1, tau.size - 1).T
    nlp.set_initial(dis_ocp["X"], Xg)
    nlp.set_initial(dis_ocp["U"], Ug)
    sol = nlp.solve()
    # return {'Xg':sol.value(dis_ocp['X']),'Ug':sol.value(dis_ocp['U']),'dtg':cs.diff(ocp.t0+tau*(ocp.tf-ocp.t0))}
    plot_solution(sol, dis_ocp, "hyper_sol")
    check_feasibility(ocp, dis_ocp, sol, "hyper_feas")


def hyper_ref_dt(guess=None):
    N = 100
    opti = cs.Opti()
    # ---- decision variables ----
    X = opti.variable(1, N + 1)
    U = opti.variable(1, N)
    # Optimize the duration of each interval directly
    dt = opti.variable(1, N)
    t0 = 0
    t = cs.hcat([t0, t0 + cs.cumsum(dt)])
    tf = 100
    opti.set_linear_scale(dt, tf / 100)

    dts = cs.SX.sym("dts")
    xs = cs.SX.sym("xs")
    us = cs.SX.sym("us")

    lag = cs.Function("lag", [xs, us], [0.5 * (xs**2 + us**2)])
    dyn = cs.Function("dyn", [xs, us], [(-(xs**3) + us)])

    def rk4():
        h = dts
        xk = xs
        uk = us

        k1 = dyn(xk, uk)
        x2 = xk + h / 2 * k1
        k2 = dyn(x2, uk)

        x3 = xk + h / 2 * k2
        k3 = dyn(x3, uk)

        x4 = xk + h * k3
        k4 = dyn(x4, uk)

        x_next = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return cs.Function("rk", [xs, us, dts], [x_next])

    def fw_euler():
        h = dts
        xk = xs
        uk = us

        k1 = dyn(xk, uk)

        x_next = xk + h * (k1)
        return cs.Function("fw_euler", [xs, us, dts], [x_next])

    opti.subject_to(X[:, 0] == 1.5)
    opti.subject_to(X[:, -1] == 1)

    ts = cs.SX.sym("ts")
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    dae = {
        "x": xs,
        "t": ts,
        "p": cs.vertcat(us, t0s, tfs),
        "ode": (tfs - t0s) * (-(xs**3) + us),
        "quad": 0.5 * (tfs - t0s) * (xs**2 + us**2),
    }
    cvodes = cs.integrator("integrator", "cvodes", dae, {"max_num_steps": 20000})
    ires = cvodes.map(N, "serial")(x0=X[:, 0:-1], p=cs.vertcat(U, t[:, 0:-1], t[:, 1:]))
    opti.subject_to(X[:, 1:] - ires["xf"] == 0)

    opti.minimize(cs.sum2(ires["qf"]))

    # ---- dynamics ----
    # obj=(lag.map(N,'serial')(X[:,0:-1],U))@dt.T
    # opti.minimize(obj,tf/N)
    # xr=rk4()
    # full=xr.map(N,'serial')(X[:,0:-1],U,dt)
    # half=xr.map(N,'serial')(xr.map(N,'serial')(X[:,0:-1],U,dt/2),U,dt/2)
    # opti.subject_to(X[:,1:]-full==0)
    # tol=1e-4
    # opti.subject_to((full-half)**2<=(15*tol)**2,225*tol**2)

    # opti.subject_to(tau[-1]==1)
    if isinstance(tf - t0, float):
        scale = tf - t0
    else:
        scale = 1
    opti.subject_to(cs.sum2(dt) == (tf - t0), scale)
    dt_min = 1e-4
    opti.subject_to(opti.bounded(dt_min, dt, tf - t0), tf - t0)

    # ---- initial guesses ----
    cgl = cs.DM([-cs.cos(k * cs.pi / N) for k in range(N + 1)]).T
    cgl01 = 0.5 * (1 + cgl)

    c1 = 0.01
    Xg = cs.hcat([1.5, cs.linspace(c1, c1, N + 1).T[:, 1:-1], 1])
    Ug = cs.linspace(c1, c1, N).T ** 3
    if guess is None:
        opti.set_initial(dt, tf / N)
        opti.set_initial(X, Xg)
        opti.set_initial(U, Ug)
    else:
        opti.set_initial(dt, guess["dtg"])
        opti.set_initial(X, guess["Xg"])
        opti.set_initial(U, guess["Ug"])

    # ---- solve ----
    opti.solver(
        "ipopt",
        {"expand": False},
        {
            "max_iter": 3000,
            "tol": 1e-5,
            "hessian_approximation": "exact",
            "nlp_scaling_method": "none",
        },
    )

    sol = opti.solve()
    print(sol.value(t))
    # ---- extract solution ----
    opt_time = np.array(sol.value(t)).flatten()
    opt_pos = np.array(sol.value(X)).flatten()
    opt_U = np.array(sol.value(U)).flatten()
    optimized_dts = np.diff(opt_time)

    print("Optimal final time:", sol.value(opti.f))

    # ---- plots ----
    #    import matplotlib.pyplot as plt
    #    plt.figure()
    #    plt.plot(opt_time, opt_pos, "o-")
    #
    #
    #    plt.figure()
    #    plt.semilogx(optimized_dts,opt_time[:-1], "o-")
    #    plt.xlabel("Time [s]")
    #    plt.ylabel("Interval duration")
    #    plt.title("Optimized Mesh Density")
    #
    #    plt.figure()
    #    plt.plot(opt_time[:-1], opt_U, "o-")
    #    plt.xlabel("Time [s]")
    #    plt.ylabel("Control")
    #    plt.title("Optimal Control")
    #    plt.show()
    return sol.value((t - t0) / (tf - t0))


def hyper_ref_dtau():
    N = 10

    opti = cs.Opti()
    # ---- decision variables ----
    X = opti.variable(1, N + 1)
    U = opti.variable(1, N)
    # Optimize the duration of each interval directly
    dtau = opti.variable(1, N)
    tau = cs.hcat([0, cs.cumsum(dtau)])
    t0 = 0
    tf = 10
    t = t0 + (tf - t0) * tau
    dt = (tf - t0) * dtau

    dts = cs.SX.sym("dts")
    xs = cs.SX.sym("xs")
    us = cs.SX.sym("us")

    lag = cs.Function("lag", [xs, us], [0.5 * (xs**2 + us**2)])
    dyn = cs.Function("dyn", [xs, us], [(-(xs**3) + us)])

    def rk4():
        h = dts
        xk = xs
        uk = us

        k1 = dyn(xk, uk)
        x2 = xk + h / 2 * k1
        k2 = dyn(x2, uk)

        x3 = xk + h / 2 * k2
        k3 = dyn(x3, uk)

        x4 = xk + h * k3
        k4 = dyn(x4, uk)

        x_next = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return cs.Function("rk", [xs, us, dts], [x_next])

    def fw_euler():
        h = dts
        xk = xs
        uk = us

        k1 = dyn(xk, uk)

        x_next = xk + h * (k1)
        return cs.Function("fw_euler", [xs, us, dts], [x_next])

    ts = cs.SX.sym("ts")
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    dae = {
        "x": xs,
        "t": ts,
        "p": cs.vertcat(us, t0s, tfs),
        "ode": (tfs - t0s) * (-(xs**3) + us),
        "quad": 0.5 * (tfs - t0s) * (xs**2 + us**2),
    }
    cvodes = cs.integrator("integrator", "cvodes", dae, {"max_num_steps": 20000})
    ires = cvodes.map(N, "serial")(x0=X[:, 0:-1], p=cs.vertcat(U, t[:, 0:-1], t[:, 1:]))
    opti.subject_to(X[:, 1:] - ires["xf"] == 0)
    opti.minimize(cs.sum2(ires["qf"]))

    # ---- dynamics ----
    # obj=(lag.map(N,'serial')(X[:,0:-1],U))@dt.T
    # opti.minimize(obj,tf/N)
    # xr=rk4()
    # full=xr.map(N,'serial')(X[:,0:-1],U,dt)
    # half=xr.map(N,'serial')(xr.map(N,'serial')(X[:,0:-1],U,dt/2),U,dt/2)
    # opti.subject_to(X[:,1:]-full==0)
    # tol=1e-2
    # opti.subject_to((full-half)**2<=(15*tol)**2,225*tol**2)

    opti.subject_to(X[:, 0] == 1.5)
    opti.subject_to(X[:, -1] == 1)
    # opti.subject_to(tau[-1]==1)
    opti.subject_to(cs.sum2(dtau) == 1)
    opti.subject_to(opti.bounded(0, dtau, 1))

    # ---- initial guesses ----
    cgl = cs.DM([-cs.cos(k * cs.pi / N) for k in range(N + 1)]).T
    cgl01 = 0.5 * (1 + cgl)
    opti.set_initial(dtau, tf / N)
    c1 = 0.1
    Xg = cs.linspace(c1, c1, N + 1).T
    Ug = cs.linspace(c1, c1, N).T ** 3
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)

    # ---- solve ----
    opti.solver(
        "ipopt",
        {"expand": False},
        {"max_iter": 3000, "tol": 1e-4, "hessian_approximation": "exact"},
    )

    sol = opti.solve()
    print(sol.value(t))
    # ---- extract solution ----
    opt_time = np.array(sol.value(t)).flatten()
    opt_pos = np.array(sol.value(X)).flatten()
    opt_U = np.array(sol.value(U)).flatten()
    optimized_dts = np.diff(opt_time)

    print("Optimal final time:", sol.value(opti.f))

    # ---- plots ----
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(opt_time, opt_pos, "b.-", label="Speed")

    plt.figure()
    plt.semilogx(optimized_dts, opt_time[:-1], "g.-")
    plt.xlabel("Time [s]")
    plt.ylabel("Interval duration")
    plt.title("Optimized Mesh Density")

    plt.figure()
    plt.plot(opt_time[:-1], opt_U, "m.-")
    plt.xlabel("Time [s]")
    plt.ylabel("Control")
    plt.title("Optimal Control")
    plt.show()
    return sol.value(tau)


def hyper_ref_dtau_fatrop():
    N = 22

    opti = cs.Opti()
    # ---- decision variables ----
    W = opti.variable(3, N)

    xnp1 = opti.variable(1)
    taunp1 = opti.variable(1)
    X = cs.hcat([W[0, :], xnp1])
    U = W[1, :]
    tau = cs.hcat([W[2, :], taunp1])
    dtau = cs.diff(tau)

    # Optimize the duration of each interval directly
    t0 = 0
    tf = 10000
    t = t0 + (tf - t0) * tau
    dt = cs.diff(t)

    dts = cs.SX.sym("dts")
    xs = cs.SX.sym("xs")
    us = cs.SX.sym("us")

    lag = cs.Function("lag", [xs, us], [0.5 * (xs**2 + us**2)])
    dyn = cs.Function("dyn", [xs, us], [(-(xs**3) + us)])

    ts = cs.SX.sym("ts")
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    dae = {
        "x": xs,
        "t": ts,
        "p": cs.vertcat(us, t0s, tfs),
        "ode": (tfs - t0s) * (-(xs**3) + us),
        "quad": 0.5 * (tfs - t0s) * (xs**2 + us**2),
    }

    cvodes = cs.integrator(
        "integrator",
        "collocation",
        dae,
        0,
        1,
        {"simplify": True, "number_of_finite_elements": 2},
    )
    ires = cvodes.map(N, "serial")(x0=X[:, 0:-1], p=cs.vertcat(U, t[:, 0:-1], t[:, 1:]))

    cons = cs.vcat(
        [
            X[:, 0] - 1.5,
            tau[0] - 0,
            cs.vec(cs.vcat([X[:, 1:] - ires["xf"], dtau])),
            X[:, -1] - 1,
            tau[-1] - 1,
        ]
    )
    lb = cs.vcat(
        [0, 0, cs.vec(cs.vcat([cs.DM.zeros(1, N), 1e-16 + cs.DM.zeros(1, N)])), 0, 0]
    )
    ub = cs.vcat([0, 0, cs.vec(cs.vcat([cs.DM.zeros(1, N), cs.DM.ones(1, N)])), 0, 0])

    opti.minimize(cs.sum2(ires["qf"]), tf)
    opti.subject_to(opti.bounded(lb, cons, ub))

    # ---- initial guesses ----
    opti.set_initial(tau, cs.linspace(0, 1, N + 1))
    c1 = 0.1
    Xg = cs.linspace(c1, c1, N + 1).T
    Ug = cs.linspace(c1, c1, N).T ** 3
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)

    # ---- solve ----
    opti.solver(
        "ipopt",
        {
            "expand": False,
        },
    )
    import matplotlib.pyplot as plt

    plt.figure()
    plt.spy(cs.hessian(opti.f, opti.x)[0].sparsity())

    sol = opti.solve()
    # ---- extract solution ----
    opt_time = np.array(sol.value(t)).flatten()
    opt_pos = np.array(sol.value(X)).flatten()
    opt_U = np.array(sol.value(U)).flatten()
    optimized_dts = np.diff(opt_time)

    print("Optimal objective :", sol.value(opti.f))

    # ---- plots ----
    plt.figure()
    plt.plot(opt_time, opt_pos, "b.-", label="Speed")

    plt.figure()
    plt.semilogx(optimized_dts, opt_time[:-1], "g.-")
    plt.xlabel("Time [s]")
    plt.ylabel("Interval duration")
    plt.title("Optimized Mesh Density")

    plt.figure()
    plt.plot(opt_time[:-1], opt_U, "m.-")
    plt.xlabel("Time [s]")
    plt.ylabel("Control")
    plt.title("Optimal Control")
    plt.show()

    return sol.value(tau)


def race_ref_dtau_fatrop():
    N = 20

    opti = cs.Opti()
    nx = 2
    nu = 1
    # ---- decision variables ----
    W = opti.variable(nx + nu + 1, N)

    xnp1 = opti.variable(nx, 1)
    tnp1 = opti.variable(1)
    X = cs.hcat([W[0:nx, :], xnp1])
    U = W[nx : nx + nu, :]
    t = cs.hcat([W[nx + nu : nx + nu + 1, :], tnp1])
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
        {"simplify": True, "number_of_finite_elements": 2},
    )
    ires = cvodes.map(N, "serial")(x0=X[:, 0:-1], p=cs.vertcat(U, t[:, 0:-1], t[:, 1:]))

    defect = X[:, 1:] - ires["xf"]

    opti.minimize(t[:, -1])
    opti.subject_to(X[:, 0] == cs.DM([0] * 2))
    opti.subject_to(X[0, -1] == 1)
    opti.subject_to(t[:, 0] == 0)
    dtn = (t[:, -1] - t[:, 0]) / N
    opti.subject_to(opti.bounded(0.1 * dtn, dt, 2 * dtn))
    opti.subject_to(defect == 0)
    opti.subject_to(opti.bounded(0, U, 1))
    opti.subject_to(
        opti.bounded(-cs.DM.inf(), X[1, :] - (1 - 0.5 * cs.sin(2 * cs.pi * X[0, :])), 0)
    )

    # ---- initial guesses ----
    Xg = cs.vertcat(cs.linspace(0, 1, N + 1).T, cs.linspace(0, 0, N + 1).T)
    Ug = cs.linspace(0, 0, N).T
    tg = cs.linspace(0, 1, N + 1).T
    opti.set_initial(t, tg)
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)

    # ---- solve ----
    opti.solver(
        "ipopt",
        {
            "expand": False,
        },
    )

    sol = opti.solve()
    # ---- post-processing        ------
    from pylab import plot, step, figure, legend, show, spy

    plot(sol.value(t), sol.value(X[1, :]), "o-", label="speed")
    plot(sol.value(t), sol.value(X[0, :]), "o-", label="pos")
    plot(
        sol.value(t),
        sol.value(1 - 0.5 * cs.sin(2 * cs.pi * X[0, :])),
        "r--",
        label="speed limit",
    )
    step(sol.value(t)[0:-1], sol.value(U), "k", label="throttle")
    legend(loc="upper left")

    figure()
    spy(sol.value(cs.jacobian(opti.g, opti.x)))
    figure()
    spy(sol.value(cs.hessian(opti.f + cs.dot(opti.lam_g, opti.g), opti.x)[0]))

    show()

    return sol.value(dt)


def moon_ref_dtau_fatrop():
    N = 25

    opti = cs.Opti()
    nx = 2
    nu = 1
    W = opti.variable(nx + nu + 1, N)

    xnp1 = opti.variable(nx, 1)
    tnp1 = opti.variable(1)
    X = cs.hcat([W[0:nx, :], xnp1])
    U = W[nx : nx + nu, :]
    t = cs.hcat([W[nx + nu : nx + nu + 1, :], tnp1])

    opti.set_linear_scale(t, 8 * cs.DM.ones(t.shape))
    opti.set_linear_scale(
        X, cs.vcat([10 * cs.DM.ones(1, N + 1), 2 * cs.DM.ones(1, N + 1)])
    )
    opti.set_linear_scale(U, 3 * cs.DM.ones(U.shape))

    dt = cs.diff(t)
    xs = cs.SX.sym("xs", nx, 1)
    us = cs.SX.sym("us", nu, 1)
    ts = cs.SX.sym("ts")
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    dae = {
        "x": xs,
        "t": ts,
        "p": cs.vertcat(us, t0s, tfs),
        "ode": (tfs - t0s) * cs.vertcat(xs[1], us - 1.5),
        "quad": (tfs - t0s) * us,
    }
    cvodes = cs.integrator(
        "integrator",
        "collocation",
        dae,
        0,
        1,
        {"simplify": True, "number_of_finite_elements": 2},
    )
    ires = cvodes.map(N, "serial")(x0=X[:, 0:-1], p=cs.vertcat(U, t[:, 0:-1], t[:, 1:]))
    defect = X[:, 1:] - ires["xf"]
    opti.minimize(cs.sum2(ires["qf"]))
    opti.subject_to(X[:, 0] == cs.DM([10, -2]))
    opti.subject_to(X[:, -1] == cs.DM([0] * 2))
    opti.subject_to(t[:, 0] == 0)
    dtn = (t[:, -1] - t[:, 0]) / N
    opti.subject_to(opti.bounded(0.1 * dtn, dt, 2 * dtn))
    opti.subject_to(defect == 0)
    opti.subject_to(opti.bounded(0, U, 3))
    Xg = cs.vertcat(cs.linspace(10, 0, N + 1).T, cs.linspace(-2, 0, N + 1).T)
    Ug = cs.linspace(0, 3, N).T
    tg = cs.linspace(0, 1, N + 1).T
    opti.set_initial(t, tg)
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)
    opti.solver(
        "ipopt",
        {
            "expand": False,
        },
    )
    sol = opti.solve()
    # ---- post-processing        ------
    from pylab import plot, step, figure, legend, show, spy

    plot(sol.value(t), sol.value(X[1, :]), "o-", label="speed")
    plot(sol.value(t), sol.value(X[0, :]), "o-", label="pos")
    step(sol.value(t)[0:-1], sol.value(U), "k", label="throttle")
    legend(loc="upper left")
    figure()
    spy(sol.value(cs.jacobian(opti.g, opti.x)))
    figure()
    spy(sol.value(cs.hessian(opti.f + cs.dot(opti.lam_g, opti.g), opti.x)[0]))
    show()
    return sol.value(dt)


def energy_ref_dtau_fatrop():
    N = 25

    opti = cs.Opti()
    nx = 2
    nu = 1
    W = opti.variable(nx + nu + 1, N)

    xnp1 = opti.variable(nx, 1)
    tnp1 = opti.variable(1)
    X = cs.hcat([W[0:nx, :], xnp1])
    U = W[nx : nx + nu, :]
    t = cs.hcat([W[nx + nu : nx + nu + 1, :], tnp1])

    #    opti.set_linear_scale(t,8*cs.DM.ones(t.shape))
    #    opti.set_linear_scale(X,cs.vcat([10*cs.DM.ones(1,N+1),2*cs.DM.ones(1,N+1)]))
    #    opti.set_linear_scale(U,3*cs.DM.ones(U.shape))

    dt = cs.diff(t)
    xs = cs.SX.sym("xs", nx, 1)
    us = cs.SX.sym("us", nu, 1)
    ts = cs.SX.sym("ts")
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    dae = {
        "x": xs,
        "t": ts,
        "p": cs.vertcat(us, t0s, tfs),
        "ode": (tfs - t0s) * cs.vertcat(xs[1], us),
        "quad": (tfs - t0s) * 0.5 * us**2,
    }
    cvodes = cs.integrator(
        "integrator",
        "collocation",
        dae,
        0,
        1,
        {"simplify": True, "number_of_finite_elements": 2},
    )
    ires = cvodes.map(N, "serial")(x0=X[:, 0:-1], p=cs.vertcat(U, t[:, 0:-1], t[:, 1:]))
    defect = X[:, 1:] - ires["xf"]
    opti.minimize(cs.sum2(ires["qf"]))
    opti.subject_to(X[:, 0] == cs.DM([0, 1]))
    opti.subject_to(X[:, -1] == cs.DM([0, -1]))
    opti.subject_to(t[:, 0] == 0)
    opti.subject_to(t[:, -1] == 1)
    dtn = (t[:, -1] - t[:, 0]) / N
    opti.subject_to(opti.bounded(0.1 * dtn, dt, 2 * dtn))
    opti.subject_to(defect == 0)
    opti.subject_to(opti.bounded(-cs.DM.inf(), X[0, :], 0.04))
    Xg = cs.vertcat(cs.linspace(0, 1, N + 1).T, cs.linspace(1, -1, N + 1).T)
    Ug = cs.linspace(0, 0, N).T
    tg = cs.linspace(0, 1, N + 1).T
    opti.set_initial(t, tg)
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)
    opti.solver(
        "ipopt",
        {
            "expand": False,
        },
    )
    sol = opti.solve()
    # ---- post-processing        ------
    from pylab import plot, step, figure, legend, show, spy

    plot(sol.value(t), sol.value(X[1, :]), "o-", label="speed")
    plot(sol.value(t), sol.value(X[0, :]), "o-", label="pos")
    step(sol.value(t)[0:-1], sol.value(U), "k", label="throttle")
    legend(loc="upper left")
    figure()
    spy(sol.value(cs.jacobian(opti.g, opti.x)))
    figure()
    spy(sol.value(cs.hessian(opti.f + cs.dot(opti.lam_g, opti.g), opti.x)[0]))
    show()
    return sol.value(dt)


def chemical_ref_dtau_fatrop():
    N = 25

    opti = cs.Opti()
    nx = 2
    nu = 1
    W = opti.variable(nx + nu + 1, N)

    xnp1 = opti.variable(nx, 1)
    tnp1 = opti.variable(1)
    X = cs.hcat([W[0:nx, :], xnp1])
    U = W[nx : nx + nu, :]
    t = cs.hcat([W[nx + nu : nx + nu + 1, :], tnp1])

    #    opti.set_linear_scale(t,8*cs.DM.ones(t.shape))
    #    opti.set_linear_scale(X,cs.vcat([10*cs.DM.ones(1,N+1),2*cs.DM.ones(1,N+1)]))
    #    opti.set_linear_scale(U,3*cs.DM.ones(U.shape))

    dt = cs.diff(t)
    xs = cs.SX.sym("xs", nx, 1)
    us = cs.SX.sym("us", nu, 1)
    ts = cs.SX.sym("ts")
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    rho = 2.5
    k = 1.5
    dae = {
        "x": xs,
        "t": ts,
        "p": cs.vertcat(us, t0s, tfs),
        "ode": (tfs - t0s) * cs.vertcat(-xs[0] * us, us * xs[0] - rho * us**k * xs[1]),
    }
    cvodes = cs.integrator(
        "integrator",
        "collocation",
        dae,
        0,
        1,
        {"simplify": True, "number_of_finite_elements": 2},
    )
    ires = cvodes.map(N, "serial")(x0=X[:, 0:-1], p=cs.vertcat(U, t[:, 0:-1], t[:, 1:]))
    defect = X[:, 1:] - ires["xf"]
    opti.minimize(-X[1, -1])
    opti.subject_to(X[:, 0] == cs.DM([1, 0.01]))
    opti.subject_to(t[:, 0] == 0)
    opti.subject_to(t[:, -1] == 2)
    dtn = (t[:, -1] - t[:, 0]) / N
    opti.subject_to(opti.bounded(0.1 * dtn, dt, 2 * dtn))
    opti.subject_to(defect == 0)
    opti.subject_to(opti.bounded(0.1, U, 0.3))
    Xg = cs.vertcat(cs.linspace(1, 0, N + 1).T, cs.linspace(0, 1, N + 1).T)
    Ug = cs.linspace(1, 1, N).T
    tg = cs.linspace(0, 2, N + 1).T
    opti.set_initial(t, tg)
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)
    opti.solver(
        "ipopt",
        {
            "expand": False,
        },
    )
    sol = opti.solve()
    # ---- post-processing        ------
    from pylab import plot, step, figure, legend, show, spy

    plot(sol.value(t), sol.value(X[1, :]), "o-", label="speed")
    plot(sol.value(t), sol.value(X[0, :]), "o-", label="pos")
    step(sol.value(t)[0:-1], sol.value(U), "k", label="throttle")
    legend(loc="upper left")
    figure()
    spy(sol.value(cs.jacobian(opti.g, opti.x)))
    figure()
    spy(sol.value(cs.hessian(opti.f + cs.dot(opti.lam_g, opti.g), opti.x)[0]))
    show()
    return sol.value(dt)


def catalyst_ref_dtau_fatrop():
    N = 100

    opti = cs.Opti()
    nx = 2
    nu = 1
    W = opti.variable(nx + nu + 1, N)

    xnp1 = opti.variable(nx, 1)
    tnp1 = opti.variable(1)
    X = cs.hcat([W[0:nx, :], xnp1])
    U = W[nx : nx + nu, :]
    t = cs.hcat([W[nx + nu : nx + nu + 1, :], tnp1])

    #    opti.set_linear_scale(t,8*cs.DM.ones(t.shape))
    #    opti.set_linear_scale(X,cs.vcat([10*cs.DM.ones(1,N+1),2*cs.DM.ones(1,N+1)]))
    #    opti.set_linear_scale(U,3*cs.DM.ones(U.shape))

    dt = cs.diff(t)
    xs = cs.SX.sym("xs", nx, 1)
    us = cs.SX.sym("us", nu, 1)
    ts = cs.SX.sym("ts")
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    rho = 2.5
    k1 = 1
    k3 = 1
    k2 = 10
    dae = {
        "x": xs,
        "t": ts,
        "p": cs.vertcat(us, t0s, tfs),
        "ode": (tfs - t0s)
        * cs.vertcat(
            -(k1 * xs[0] - k2 * xs[1]) * us,
            us * (k1 * xs[0] - k2 * xs[1]) - (1 - us) * k3 * xs[1],
        ),
    }
    cvodes = cs.integrator(
        "integrator",
        "collocation",
        dae,
        0,
        1,
        {"simplify": True, "number_of_finite_elements": 2},
    )
    ires = cvodes.map(N, "serial")(x0=X[:, 0:-1], p=cs.vertcat(U, t[:, 0:-1], t[:, 1:]))
    defect = X[:, 1:] - ires["xf"]
    opti.minimize(-(1-X[1, -1]-X[0,-1]))
    opti.subject_to(X[:, 0] == cs.DM([1, 0]))
    opti.subject_to(t[:, 0] == 0)
    opti.subject_to(t[:, -1] == 4)
    dtn = (t[:, -1] - t[:, 0]) / N
    opti.subject_to(opti.bounded(0.1 * dtn, dt, 2 * dtn))
    opti.subject_to(defect == 0)
    opti.subject_to(opti.bounded(0, U, 1))
    Xg = cs.vertcat(cs.linspace(1, 0, N + 1).T, cs.linspace(0, 1, N + 1).T)
    Ug = cs.linspace(0, 1, N).T
    tg = cs.linspace(0, 4, N + 1).T
    opti.set_initial(t, tg)
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)
    opti.solver(
        "ipopt",
        {
            "expand": False,
        },
    )
    sol = opti.solve()
    # ---- post-processing        ------
    from pylab import plot, step, figure, legend, show, spy

    plot(sol.value(t), sol.value(X[1, :]), "o-", label="speed")
    plot(sol.value(t), sol.value(X[0, :]), "o-", label="pos")
    step(sol.value(t)[0:-1], sol.value(U), "k", label="throttle")
    legend(loc="upper left")
    figure()
    spy(sol.value(cs.jacobian(opti.g, opti.x)))
    figure()
    spy(sol.value(cs.hessian(opti.f + cs.dot(opti.lam_g, opti.g), opti.x)[0]))
    show()
    return sol.value(dt)

def ally_ref_dtau_fatrop():
    N = 100

    opti = cs.Opti()
    nx = 3
    nu = 1
    W = opti.variable(nx + nu + 1, N)

    xnp1 = opti.variable(nx, 1)
    tnp1 = opti.variable(1)
    X = cs.hcat([W[0:nx, :], xnp1])
    U = W[nx : nx + nu, :]
    t = cs.hcat([W[nx + nu : nx + nu + 1, :], tnp1])

    #    opti.set_linear_scale(t,8*cs.DM.ones(t.shape))
    #    opti.set_linear_scale(X,cs.vcat([10*cs.DM.ones(1,N+1),2*cs.DM.ones(1,N+1)]))
    #    opti.set_linear_scale(U,3*cs.DM.ones(U.shape))

    dt = cs.diff(t)
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
            us,
            0.5*(xs[1]**2-xs[0]**2)
        ),
    }
    cvodes = cs.integrator(
        "integrator",
        "collocation",
        dae,
        0,
        1,
        {"simplify": True, "number_of_finite_elements": 2},
    )
    ires = cvodes.map(N, "serial")(x0=X[:, 0:-1], p=cs.vertcat(U, t[:, 0:-1], t[:, 1:]))
    defect = X[:, 1:] - ires["xf"]
    opti.minimize(X[2,-1])
    opti.subject_to(X[:, 0] == cs.DM([0, 1, 0]))
    opti.subject_to(t[:, 0] == 0)
    opti.subject_to(t[:, -1] == cs.pi/2)
    dtn = (t[:, -1] - t[:, 0]) / N
    opti.subject_to(opti.bounded(0.1 * dtn, dt, 2 * dtn))
    opti.subject_to(defect == 0)
    opti.subject_to(opti.bounded(-1, U, 1))
    Xg = cs.vertcat(cs.linspace(0, 1, N + 1).T, cs.linspace(1, 0, N + 1).T,cs.linspace(0, 1, N + 1).T)
    Ug = cs.linspace(-1, 1, N).T
    tg = cs.linspace(0, cs.pi/2, N + 1).T
    opti.set_initial(t, tg)
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)
    opti.solver(
        "ipopt",
        {
            "expand": False,
        },
    )
    sol = opti.solve()
    # ---- post-processing        ------
    from pylab import plot, step, figure, legend, show, spy

    plot(sol.value(t), sol.value(X[1, :]), "o-", label="speed")
    plot(sol.value(t), sol.value(X[0, :]), "o-", label="pos")
    step(sol.value(t)[0:-1], sol.value(U), "k", label="throttle")
    legend(loc="upper left")
    figure()
    spy(sol.value(cs.jacobian(opti.g, opti.x)))
    figure()
    spy(sol.value(cs.hessian(opti.f + cs.dot(opti.lam_g, opti.g), opti.x)[0]))
    show()
    return sol.value(dt)


def jenning_ref_dtau_fatrop():
    N = 25

    opti = cs.Opti()
    nx = 3
    nu = 1
    W = opti.variable(nx + nu + 1, N)

    xnp1 = opti.variable(nx, 1)
    tnp1 = opti.variable(1)
    X = cs.hcat([W[0:nx, :], xnp1])
    U = W[nx : nx + nu, :]
    t = cs.hcat([W[nx + nu : nx + nu + 1, :], tnp1])

    #    opti.set_linear_scale(t,8*cs.DM.ones(t.shape))
    #    opti.set_linear_scale(X,cs.vcat([10*cs.DM.ones(1,N+1),2*cs.DM.ones(1,N+1)]))
    #    opti.set_linear_scale(U,3*cs.DM.ones(U.shape))

    dt = cs.diff(t)
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
            us,
            cs.cos(xs[0]),
            cs.sin(xs[0]),
        ),
    }
    cvodes = cs.integrator(
        "integrator",
        "collocation",
        dae,
        0,
        1,
        {"simplify": True, "number_of_finite_elements": 2},
    )
    ires = cvodes.map(N, "serial")(x0=X[:, 0:-1], p=cs.vertcat(U, t[:, 0:-1], t[:, 1:]))
    defect = X[:, 1:] - ires["xf"]
    opti.minimize(t[:,-1],50)
    opti.subject_to(X[:, 0] == cs.DM([cs.pi/2, 4, 0]))
    opti.subject_to(X[1:3, -1] == cs.DM([0, 0]))
    opti.subject_to(t[:, 0] == 0)
    dtn = (t[:, -1] - t[:, 0]) / N
    opti.subject_to(opti.bounded(0.1 * dtn, dt, 2 * dtn))
    opti.subject_to(defect == 0)
    opti.subject_to(opti.bounded(-2, U, 2))
    Xg = cs.vertcat(cs.linspace(cs.pi/2, 1, N + 1).T, cs.linspace(4, 0, N + 1).T,cs.linspace(0, 1, N + 1).T)
    Ug = cs.linspace(-2, 2, N).T
    tg = cs.linspace(0, cs.pi/2, N + 1).T
    opti.set_initial(t, tg)
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)
    opti.solver(
        "ipopt",
        {
            "expand": False,
        },
    )
    sol = opti.solve()
    # ---- post-processing        ------
    from pylab import plot, step, figure, legend, show, spy

    plot(sol.value(t), sol.value(X[1, :]), "o-", label="speed")
    plot(sol.value(t), sol.value(X[0, :]), "o-", label="pos")
    step(sol.value(t)[0:-1], sol.value(U), "k", label="throttle")
    legend(loc="upper left")
    figure()
    spy(sol.value(cs.jacobian(opti.g, opti.x)))
    figure()
    spy(sol.value(cs.hessian(opti.f + cs.dot(opti.lam_g, opti.g), opti.x)[0]))
    show()
    return sol.value(dt)


def bressnan_ref_dtau_fatrop():
    N = 100

    opti = cs.Opti()
    nx = 3
    nu = 1
    W = opti.variable(nx + nu + 1, N)

    xnp1 = opti.variable(nx, 1)
    tnp1 = opti.variable(1)
    X = cs.hcat([W[0:nx, :], xnp1])
    U = W[nx : nx + nu, :]
    t = cs.hcat([W[nx + nu : nx + nu + 1, :], tnp1])

    #opti.set_linear_scale(t,10*cs.DM.ones(t.shape))
    #opti.set_linear_scale(X,cs.vcat([cs.DM.ones(1,N+1),10*cs.DM.ones(1,N+1),100*cs.DM.ones(1,N+1)]))
    #opti.set_linear_scale(U,1*cs.DM.ones(U.shape))

    dt = cs.diff(t)
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
            us,
            -xs[0],
            xs[1]-xs[0]**2,
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
    opti.minimize(-X[2,-1])
    opti.subject_to(X[:, 0] == cs.DM([0]*3))
    opti.subject_to(t[:, 0] == 0)
    opti.subject_to(t[:, -1] == 10)
    dtn = (t[:, -1] - t[:, 0]) / N
    opti.subject_to(opti.bounded(0.1 * dtn, dt, 2 * dtn))
    opti.subject_to(defect == 0)
    opti.subject_to(opti.bounded(-1, U, 1))
    Xg = cs.vertcat(cs.linspace(0, 0, N + 1).T, cs.linspace(0, 2, N + 1).T,cs.linspace(0, 5, N + 1).T)
    Ug = cs.linspace(0, 0, N).T
    tg = cs.linspace(0, 10, N + 1).T
    opti.set_initial(t, tg)
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)
    opti.solver(
        "ipopt",
        {
            "expand": False,
        },
    )
    sol = opti.solve()
    # ---- post-processing        ------
    from pylab import plot, step, figure, legend, show, spy

    plot(sol.value(t), sol.value(X[0, :]), "o-", label="z1")
    plot(sol.value(t), sol.value(X[1, :])/10, "o-", label="z2")
    plot(sol.value(t), sol.value(X[2, :])/50, "o-", label="z3")
    step(sol.value(t)[0:-1], sol.value(U), "k", label="throttle")
    legend(loc="upper left")
    figure()
    spy(sol.value(cs.jacobian(opti.g, opti.x)))
    figure()
    spy(sol.value(cs.hessian(opti.f + cs.dot(opti.lam_g, opti.g), opti.x)[0]))
    show()
    return sol.value(dt)

def solve_fedbatch_bioprocess():
    N = 100
    tf = 10.0  # Final time

    opti = cs.Opti()
    nx = 7
    nu = 2
    W = opti.variable(nx + nu+1, N)

    # State variables at the final node N
    xnp1 = opti.variable(nx, 1)
    tnp1= opti.variable(1, 1)

    X = cs.hcat([W[0:nx, :], xnp1])
    U = W[nx : nx + nu, :]
    t= cs.hcat([W[nx+nu:nx+nu+1, :], tnp1])

    # Time vector (fixed final time tf)
    t = cs.linspace(0, tf, N + 1)
    dt = tf / N

    # Constants
    c1 = 100.0
    c2 = 0.51
    c3 = 4.0

    # Symbolic variables for DAE formulation
    xs = cs.SX.sym("xs", nx, 1)
    us = cs.SX.sym("us", nu, 1)
    ts = cs.SX.sym("ts")

    # Intermediate algebraic equations & growth rates
    x1, x2, x3, x4, x5, x6, x7 = (
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        xs[4],
        xs[5],
        xs[6],
    )
    u1, u2 = us[0], us[1]

    t1 = 14.35 + x3 + (x3**2) / 111.5
    t2 = 0.22 + x5
    t3 = (0.22 * x7) / t2 + x6

    g1 = (x3 * t3) / t1
    g2 = (0.233 * x3 * (0.0005 + x5)) / ((0.022 + x5) * t1)
    g3 = (0.09 * x5) / (0.034 + x5)

    ode = cs.vertcat(
        u1 + u2,
        g1 * x2 - ((u1 + u2) * x2) / x1,
        (u1 / x1) * c1 - ((u1 + u2) * x3) / x1 - (g1 * x2) / c2,
        g2 * x2 - ((u1 + u2) * x4) / x1,
        (u2 * c3) / x1 - ((u1 + x2) * x5) / x1,  # wait: note u2*c3/x1 - (u1+u2)*x5/x1
        -g3 * x6,
        g3 * (1 - x7),
    )

    dae = {
        "x": xs,
        "t": ts,
        "p": us,
        "ode": ode,
    }

    # Integrator using direct collocation
    cvodes = cs.integrator(
        "integrator",
        "collocation",
        dae,
        0,
        dt,
        {"simplify": True, "number_of_finite_elements": 2},
    )

    ires = cvodes.map(N, "serial")(x0=X[:, 0:-1], p=U)
    defect = X[:, 1:] - ires["xf"]

    # Objective function: min -x4(tf) * x1(tf)
    opti.minimize(-X[3, -1] * X[0, -1])

    # Initial conditions
    opti.subject_to(
        X[:, 0]
        == cs.DM(
            [
                1.0,  # x1(0) = 1
                0.1,  # x2(0) = 0.1
                40.0,  # x3(0) = 40
                0.0,  # x4(0) = 0
                0.0,  # x5(0) = 0
                1.0,  # x6(0) = 1
                0.0,  # x7(0) = 0
            ]
        )
    )

    # Collocation continuity constraints
    opti.subject_to(defect == 0)

    # Control bounds: 0 <= u1, u2 <= 1
    opti.subject_to(opti.bounded(0, U, 1))

    # Initial guesses
    opti.set_initial(X, cs.repmat(cs.DM([1, 0.1, 40, 0, 0, 1, 0]), 1, N + 1))
    opti.set_initial(U, 0.5 * cs.DM.ones(nu, N))

    opti.solver(
        "ipopt",
        {
            "expand": True,
        },
    )

    sol = opti.solve()

    # ---- Post-processing ------
    from pylab import figure, legend, plot, show, step

    t_val = sol.value(t)
    X_val = sol.value(X)
    U_val = sol.value(U)

    figure(figsize=(10, 6))
    for i in range(nx):
        plot(t_val, X_val[i, :], label=f"$x_{i+1}$")
    legend()
    figure(figsize=(10, 4))
    step(t_val[:-1], U_val[0, :], label="$u_1$", where="post")
    step(t_val[:-1], U_val[1, :], label="$u_2$", where="post")
    legend()
    show()

    return sol.value(-X[3, -1] * X[0, -1])

def solve_fedbatch_variable_grid():
    N = 30

    opti = cs.Opti()
    nx = 7
    nu = 2
    W = opti.variable(nx + nu + 1, N)

    xnp1 = opti.variable(nx, 1)
    tnp1 = opti.variable(1)
    X = cs.hcat([W[0:nx, :], xnp1])
    U = W[nx : nx + nu, :]
    t = cs.hcat([W[nx + nu : nx + nu + 1, :], tnp1])

    # Linear scaling for stability
    #opti.set_linear_scale(t, 10 * cs.DM.ones(t.shape))
    #opti.set_linear_scale(
    #    X,
    #    cs.vcat(
    #        [
    #            cs.DM.ones(1, N + 1),
    #            cs.DM.ones(1, N + 1),
    #            10 * cs.DM.ones(1, N + 1),
    #            cs.DM.ones(1, N + 1),
    #            cs.DM.ones(1, N + 1),
    #            cs.DM.ones(1, N + 1),
    #            cs.DM.ones(1, N + 1),
    #        ]
    #    ),
    #)
    #opti.set_linear_scale(U, cs.DM.ones(U.shape))

    dt = cs.diff(t)
    xs = cs.SX.sym("xs", nx, 1)
    us = cs.SX.sym("us", nu, 1)
    ts = cs.SX.sym("ts")
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")

    c1 = 100.0
    c2 = 0.51
    c3 = 4.0

    x1, x2, x3, x4, x5, x6, x7 = (
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        xs[4],
        xs[5],
        xs[6],
    )
    u1, u2 = us[0], us[1]

    t1 = 14.35 + x3 + (x3**2) / 111.5
    t2 = 0.22 + x5
    t3 = (0.22 * x7) / t2 + x6

    g1 = (x3 * t3) / t1
    g2 = (0.233 * x3 * (0.0005 + x5)) / ((0.022 + x5) * t1)
    g3 = (0.09 * x5) / (0.034 + x5)

    dae = {
        "x": xs,
        "t": ts,
        "p": cs.vertcat(us, t0s, tfs),
        "ode": (tfs - t0s)
        * cs.vertcat(
            u1 + u2,
            g1 * x2 - ((u1 + u2) * x2) / x1,
            (u1 / x1) * c1 - ((u1 + u2) * x3) / x1 - (g1 * x2) / c2,
            g2 * x2 - ((u1 + u2) * x4) / x1,
            (u2 * c3) / x1 - ((u1 + u2) * x5) / x1,
            -g3 * x6,
            g3 * (1 - x7),
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

    opti.minimize(-X[3, -1] * X[0, -1])

    opti.subject_to(
        X[:, 0] == cs.DM([1.0, 0.1, 40.0, 0.0, 0.0, 1.0, 0.0])
    )
    opti.subject_to(t[:, 0] == 0)
    opti.subject_to(t[:, -1] == 10.0)

    dtn = (t[:, -1] - t[:, 0]) / N
    opti.subject_to(opti.bounded(0.1 * dtn, dt, 2 * dtn))
    defs=cs.repmat(cs.DM([1.0, 1, 10.0, 1.0, 1.0, 1.0, 1.0]),1,N)
    opti.subject_to(defect == 0)
    opti.subject_to(opti.bounded(0, U, 1))
    opti.subject_to(X[0,:]>=0)
    opti.subject_to(X[4,:]>=0)

    x0=cs.DM([1.0, 0.1, 40.0, 0.0, 0.0, 1.0, 0.0])
    Xg = cs.repmat(x0,1,N+1)
    Ug = cs.vcat([cs.linspace(1,1,N).T]*2)
    tg = cs.linspace(0, 10.0, N + 1).T

    opti.set_initial(t, tg)
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)

    opti.solver(
        "ipopt",
    )

    sol = opti.solve()

    from pylab import plot, show, figure, legend, step

    t_val = sol.value(t)
    X_val = sol.value(X)
    U_val = sol.value(U)

    figure(figsize=(10, 6))
    for i in range(nx):
        plot(t_val, X_val[i, :], '-o',label=f"$x_{i+1}$")
    legend()
    figure(figsize=(10, 4))
    step(t_val[:-1], U_val[0, :], label="$u_1$", where="post")
    step(t_val[:-1], U_val[1, :], label="$u_2$", where="post")
    legend()
    show()

    return sol.value(-X[3, -1] * X[0, -1])

def moon():
    ocp_data = {}
    ocp_data["nx"] = 4
    ocp_data["nu"] = 2
    ts = cs.SX.sym("ts", 1, 1)
    xs = cs.SX.sym("xs", ocp_data["nx"], 1)
    us = cs.SX.sym("us", ocp_data["nu"], 1)
    t0s = cs.SX.sym("t0s", 1, 1)
    tfs = cs.SX.sym("tfs", 1, 1)
    x0s = cs.SX.sym("x0s", ocp_data["nx"], 1)
    xfs = cs.SX.sym("xfs", ocp_data["nx"], 1)
    path = [ts, xs, us]
    event = [t0s, tfs, x0s, xfs]
    ocp_data["L"] = None
    ocp_data["E"] = cs.Function('E',event,[xfs[2]])
    ocp_data["e"] = cs.Function("e", event, [cs.vertcat(x0s - cs.DM([10,-2,0,0]), xfs[0:2,:] - cs.DM([0]*2))])
    ocp_data["e_lb"] = cs.DM([0] * 6)
    ocp_data["e_ub"] = cs.DM([0] * 6)
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = cs.DM(1)
    ocp_data["f"] = cs.Function("f", path, [cs.vertcat(us[1]*xs[1],us[1]*(us[0]-1.5),us[1]*(us[0]),us[1])])
    ocp_data['x_lb']=cs.DM([0,-cs.DM.inf(),-cs.DM.inf(),0])
    ocp_data['x_ub']=cs.DM([cs.DM.inf(),0,cs.DM.inf(),100])
    n=25
    ocp_data['u_lb']=cs.DM([0,0.825])
    ocp_data['u_ub']=cs.DM([3,8.26])

    ocp = Ocp(**ocp_data)
    nlp = cs.Opti()
 
    method = casadi_integrator(
        plugin="collocation",
        plugin_options={"number_of_finite_elements": 1, "simplify": True},
        nx=ocp.nx,
        nu=ocp.nu,
        path=[ts, xs, us],
        f=ocp.f(ts, xs, us),
    )
    disc = disc_opt(n=n, method=method)
    dis_ocp = generate_NLP(ocp, nlp, disc)
    nlp.solver("ipopt")
    Xg = cs.repmat(cs.DM([10,-2,0,0]),1,disc.n+1)
    Ug = cs.vertcat(cs.linspace(0, 3, n).T,cs.linspace(0.1, 1, n).T) 
    nlp.set_initial(dis_ocp["X"], Xg)
    nlp.set_initial(dis_ocp["U"], Ug)
    sol = nlp.solve()
    # return {'Xg':sol.value(dis_ocp['X']),'Ug':sol.value(dis_ocp['U']),'dtg':cs.diff(ocp.t0+tau*(ocp.tf-ocp.t0))}
    plot_solution(sol, dis_ocp, "moon_sol")
    check_feasibility(ocp, dis_ocp, sol, "moon_feas")


def hyper_pytrajopt_adapt():
    tfinal=10000
    ocp_data = {}
    ocp_data["nx"] = 2
    ocp_data["nu"] = 2
    ts = cs.SX.sym("ts", 1, 1)
    xs = cs.SX.sym("xs", ocp_data["nx"], 1)
    us = cs.SX.sym("us", ocp_data["nu"], 1)
    t0s = cs.SX.sym("t0s", 1, 1)
    tfs = cs.SX.sym("tfs", 1, 1)
    x0s = cs.SX.sym("x0s", ocp_data["nx"], 1)
    xfs = cs.SX.sym("xfs", ocp_data["nx"], 1)
    path = [ts, xs, us]
    event = [t0s, tfs, x0s, xfs]
    ocp_data["L"] = cs.Function("L", path, [0.5 * (xs[0]**2 + us[0]**2)*us[1]])
    ocp_data["E"] = None
    ocp_data["e"] = cs.Function("e", event, [cs.vertcat(x0s - cs.DM([1.5,0]), xfs[0] - 1,xfs[1]-tfinal)])
    ocp_data["e_lb"] = cs.DM([0] * 4)
    ocp_data["e_ub"] = cs.DM([0] * 4)
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = cs.DM(10)
    ocp_data["f"] = cs.Function("f", path, [cs.vcat([(-(xs[0]**3) + us[0])*us[1],us[1]])])
    inf=cs.DM.inf()
    ocp_data['u_lb']=cs.DM([-inf,0])
    ocp_data['u_ub']=cs.DM([inf,inf])
    #ocp_data['J_s']=cs.DM(tfinal)
    #ocp_data['x_s']=cs.DM([tfinal,tfinal,tfinal])
    #ocp_data['u_s']=cs.DM([1,tfinal])
    #ocp_data['e_s']=cs.DM([1,1,tfinal,1,tfinal])
    


    ocp = Ocp(**ocp_data)
    nlp = cs.Opti()
    method = casadi_integrator(
        plugin="cvodes",
#        plugin_options={"number_of_finite_elements": 1, "simplify": True},
        nx=ocp.nx,
        nu=ocp.nu,
        path=[ts, xs, us],
        f=ocp.f(ts, xs, us),
    )
    n=100
    disc = disc_opt(n=n, method=method)
    dis_ocp = generate_NLP(ocp, nlp, disc)
    nlp.solver("ipopt",{ "ipopt.bound_relax_factor": 0.0,"ipopt.honor_original_bounds": "yes",})

    c1 = 0.1
    u0 = c1**3
    qdot = 0.5 * (c1**2 + u0**2)

    sigma = cs.linspace(0, 1, n+1).T

    guess=hyper_pytrajopt_tau()

    Xg = cs.vertcat(
        cs.linspace(1.5,1,n+1).T,
        tfinal * sigma,
    )

    Ug = cs.vertcat(
        cs.linspace(1.5,1,n).T**3,
        tfinal * cs.DM.ones(1, n),
    )



    nlp.set_initial(dis_ocp["X"], Xg)
    nlp.set_initial(dis_ocp["U"], Ug)
    sol = nlp.solve()
    # return {'Xg':sol.value(dis_ocp['X']),'Ug':sol.value(dis_ocp['U']),'dtg':cs.diff(ocp.t0+tau*(ocp.tf-ocp.t0))}
    plot_solution(sol, dis_ocp, "hyper_sol")
    check_feasibility(ocp, dis_ocp, sol, "hyper_feas")


def hyper_pytrajopt_adapt_ross():
    tfinal=10000
    ocp_data = {}
    ocp_data["nx"] = 1
    ocp_data["nu"] = 1
    ts = cs.SX.sym("ts", 1, 1)
    xs = cs.SX.sym("xs", ocp_data["nx"], 1)
    us = cs.SX.sym("us", ocp_data["nu"], 1)
    t0s = cs.SX.sym("t0s", 1, 1)
    tfs = cs.SX.sym("tfs", 1, 1)
    x0s = cs.SX.sym("x0s", ocp_data["nx"], 1)
    xfs = cs.SX.sym("xfs", ocp_data["nx"], 1)
    path = [ts, xs, us]
    event = [t0s, tfs, x0s, xfs]
    ocp_data["L"] = cs.Function("L", path, [0.5 * (xs**2 + us**2)*0.1])
    ocp_data["E"] = None
    ocp_data["e"] = cs.Function("e", event, [cs.vertcat(x0s - cs.DM([1.5]), xfs - 1)])
    ocp_data["e_lb"] = cs.DM([0] * 2)
    ocp_data["e_ub"] = cs.DM([0] * 2)
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = cs.DM(1)
    ocp_data["f"] = cs.Function("f", path, [1000*(-(xs**3) + us)])
    #inf=cs.DM.inf()
    #ocp_data['u_lb']=cs.DM([-inf,0])
    #ocp_data['u_ub']=cs.DM([inf,inf])
    #ocp_data['J_s']=cs.DM(tfinal)
    #ocp_data['x_s']=cs.DM([tfinal,tfinal,tfinal])
    #ocp_data['u_s']=cs.DM([1,tfinal])
    #ocp_data['e_s']=cs.DM([1,1,tfinal,1,tfinal])
    


    ocp = Ocp(**ocp_data)
    nlp = cs.Opti()
    method = casadi_integrator(
        plugin="collocation",
        plugin_options={"number_of_finite_elements": 1, "simplify": True},
        nx=ocp.nx,
        nu=ocp.nu,
        path=[ts, xs, us],
        f=ocp.f(ts, xs, us),
    )
    n=25
    disc = disc_opt(n=n, method=method)
    from pytrajopt import DynamicMesh
    mesh=DynamicMesh(k1=0.001,k2=1e32)
    dis_ocp = generate_NLP(ocp, nlp, disc,mesh)
    nlp.solver("ipopt",{ "ipopt.bound_relax_factor": 0.0,"ipopt.honor_original_bounds": "yes",})



    Xg = cs.vertcat(
        cs.linspace(1.5,1,n+1).T,
    )

    Ug = cs.vertcat(
        cs.linspace(1.5,1,n).T**3,
    )


    nlp.set_initial(dis_ocp["t"], cs.linspace(0,10,n+1).T)
    nlp.set_initial(dis_ocp["X"], Xg)
    nlp.set_initial(dis_ocp["U"], Ug)
    sol = nlp.solve()
    # return {'Xg':sol.value(dis_ocp['X']),'Ug':sol.value(dis_ocp['U']),'dtg':cs.diff(ocp.t0+tau*(ocp.tf-ocp.t0))}
    plot_solution(sol, dis_ocp, "hyper_sol")
    check_feasibility(ocp, dis_ocp, sol, "hyper_feas")


def hyper_pytrajopt_tau():
    tfinal=10000
    ocp_data = {}
    ocp_data["nx"] = 1
    ocp_data["nu"] = 1
    ts = cs.SX.sym("ts", 1, 1)
    xs = cs.SX.sym("xs", ocp_data["nx"], 1)
    us = cs.SX.sym("us", ocp_data["nu"], 1)
    t0s = cs.SX.sym("t0s", 1, 1)
    tfs = cs.SX.sym("tfs", 1, 1)
    x0s = cs.SX.sym("x0s", ocp_data["nx"], 1)
    xfs = cs.SX.sym("xfs", ocp_data["nx"], 1)
    path = [ts, xs, us]
    event = [t0s, tfs, x0s, xfs]
    ocp_data["L"] = cs.Function("L", path, [0.5 *tfinal* (xs**2 + us**2)])
    ocp_data["E"] = None
    ocp_data["e"] = cs.Function("e", event, [cs.vertcat(x0s -1.5, xfs[0] - 1)])
    ocp_data["e_lb"] = cs.DM([0] * 2)
    ocp_data["e_ub"] = cs.DM([0] * 2)
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = cs.DM(1)
    ocp_data["f"] = cs.Function("f", path, [tfinal*(-(xs**3) + us)])


    ocp = Ocp(**ocp_data)
    nlp = cs.Opti()
    method = casadi_integrator(
        plugin="collocation",
        plugin_options={"number_of_finite_elements": 1, "simplify": True},
        nx=ocp.nx,
        nu=ocp.nu,
        path=[ts, xs, us],
        f=ocp.f(ts, xs, us),
    )
    n=50
    disc = disc_opt(n=n, method=method)
    dis_ocp = generate_NLP(ocp, nlp, disc)
    nlp.solver("ipopt")

    c1 = 0.1
    u0 = c1**3

    Xg = cs.vertcat(
        c1 * cs.DM.ones(1, n+1),
    )

    Ug = cs.vertcat(
        u0 * cs.DM.ones(1, n),
    )

    nlp.set_initial(dis_ocp["X"], Xg)
    nlp.set_initial(dis_ocp["U"], Ug)
    sol = nlp.solve()
    # return {'Xg':sol.value(dis_ocp['X']),'Ug':sol.value(dis_ocp['U']),'dtg':cs.diff(ocp.t0+tau*(ocp.tf-ocp.t0))}
    plot_solution(sol, dis_ocp, "hyper_sol")
    check_feasibility(ocp, dis_ocp, sol, "hyper_feas")
    return {'Xg':sol.value(dis_ocp['X']),'Ug':sol.value(dis_ocp['U'])}




def hyper_ref_dtau():
    N = 100

    opti = cs.Opti()
    # ---- decision variables ----
    X = opti.variable(1, N + 1)
    U = opti.variable(1, N)
    # Optimize the duration of each interval directly
    dtau = opti.variable(1, N)
    tau=cs.hcat([0,cs.cumsum(dtau)])
    t0=0
    tf=10000
    t=t0+(tf-t0)*tau
    dt=(tf-t0)*dtau

    dts=cs.SX.sym('dts')
    xs=cs.SX.sym('xs')
    us=cs.SX.sym('us')

    lag=cs.Function('lag',[xs,us],[0.5*(xs**2+us**2)])
    dyn=cs.Function('dyn',[xs,us],[(-xs**3+us)])


    def rk4():
        h = dts
        xk = xs
        uk = us

        k1 = dyn(xk, uk)
        x2 = xk + h / 2 * k1
        k2 = dyn(x2, uk)

        x3 = xk + h / 2 * k2
        k3 = dyn(x3, uk)

        x4 = xk + h * k3
        k4 = dyn(x4, uk)

        x_next = xk + h / 6 * (k1 + 2*k2 + 2*k3 + k4)
        return cs.Function('rk',[xs,us,dts],[x_next])

    def fw_euler():
        h = dts
        xk = xs
        uk = us

        k1 = dyn(xk, uk)

        x_next = xk + h  * (k1 )
        return cs.Function('fw_euler',[xs,us,dts],[x_next])    

    ts=cs.SX.sym('ts')
    t0s=cs.SX.sym('t0s')
    tfs=cs.SX.sym('tfs')
    dae = {'x': xs, 't':ts,'p': cs.vertcat(us,t0s,tfs), 'ode': (tfs-t0s)*(-xs**3+us),'quad':0.5*(tfs-t0s)*(xs**2+us**2)}
    cvodes = cs.integrator('integrator', 'cvodes', dae,{'max_num_steps': 20000})
    ires=cvodes.map(N,'serial')(x0=X[:,0:-1],p=cs.vertcat(U,t[:,0:-1],t[:,1:]))
    opti.subject_to(X[:,1:]-ires['xf']==0)
    opti.minimize(cs.sum2(ires['qf']))    



    # ---- dynamics ----
    #obj=(lag.map(N,'serial')(X[:,0:-1],U))@dt.T
    #opti.minimize(obj,tf/N)
    #xr=rk4()
    #full=xr.map(N,'serial')(X[:,0:-1],U,dt)
    #half=xr.map(N,'serial')(xr.map(N,'serial')(X[:,0:-1],U,dt/2),U,dt/2)
    #opti.subject_to(X[:,1:]-full==0)
    #tol=1e-2
    #opti.subject_to((full-half)**2<=(15*tol)**2,225*tol**2)
    




    opti.subject_to(X[:,0] == 1.5)
    opti.subject_to(X[:,-1] == 1)
    #opti.subject_to(tau[-1]==1)
    opti.subject_to(cs.sum2(dtau)==1)
    opti.subject_to(opti.bounded(0,dtau,1))

    # ---- initial guesses ----
    cgl=cs.DM([-cs.cos(k*cs.pi/N) for k in range(N+1)]).T
    cgl01=0.5*(1+cgl)
    opti.set_initial(dtau, tf/N)
    c1=0.1
    Xg=cs.linspace(c1,c1,N+1).T
    Ug=cs.linspace(c1,c1,N).T**3
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)

    # ---- solve ----
    opti.solver(
        "ipopt",
        {"expand": False},
        {
            "max_iter": 3000,
            "tol": 1e-4,
            "hessian_approximation": 'exact'
        }
    )

    sol = opti.solve()
    print(sol.value(t))
    # ---- extract solution ----
    opt_time = np.array(sol.value(t)).flatten()
    opt_pos = np.array(sol.value(X)).flatten()
    opt_U = np.array(sol.value(U)).flatten()
    optimized_dts = np.diff(opt_time)

    print("Optimal final time:", sol.value(opti.f))


    # ---- plots ----
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(opt_time, opt_pos, "b.-", label="Speed")


    plt.figure()
    plt.semilogx(optimized_dts,opt_time[:-1], "g.-")
    plt.xlabel("Time [s]")
    plt.ylabel("Interval duration")
    plt.title("Optimized Mesh Density")

    plt.figure()
    plt.plot(opt_time[:-1], opt_U, "m.-")
    plt.xlabel("Time [s]")
    plt.ylabel("Control")
    plt.title("Optimal Control")
    plt.show()
    return sol.value(tau)




if __name__ == "__main__":
    #print(moon())
    #moon_ref_dtau_fatrop()
    #hyper_pytrajopt_tau()
    hyper_pytrajopt_adapt_ross()
    #hyper_ref_dtau_fatrop()
