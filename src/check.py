import genesis as gs

gs.init(backend=gs.gpu, logging_level="warning")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
    rigid_options=gs.options.RigidOptions(
        dt=0.02,
        constraint_solver=gs.constraint_solver.Newton,
        enable_collision=True,
        enable_joint_limit=True,
    ),
    show_viewer=True,
)

scene.add_entity(gs.morphs.Plane())

robot = scene.add_entity(
    gs.morphs.URDF(
        file="urdf/go2/urdf/go2.urdf",
        pos=(0.0, 0.0, 0.42),
        quat=(1.0, 0.0, 0.0, 0.0),
    )
)

scene.build(n_envs=1)

dof_names = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

print("============================================================")
print("GO2 DOF CHECK")
print("============================================================")

motor_dofs = []
for name in dof_names:
    joint = robot.get_joint(name)
    idx = joint.dof_idx_local
    motor_dofs.append(idx)
    print(f"{name:20s} -> {idx}")

print("motor_dofs =", motor_dofs)
print("unique =", sorted(set(motor_dofs)))

if len(set(motor_dofs)) != 12:
    raise RuntimeError("DOF indices are not unique. Joint mapping is broken.")

print("OK: all 12 joints found.")

for _ in range(300):
    scene.step()