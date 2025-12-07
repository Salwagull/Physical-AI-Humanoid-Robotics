# Simulation Diagrams

This directory contains diagrams and visual aids for robot simulation concepts covered in Chapters 4-5.

## Diagrams Needed:

### Chapter 4: Gazebo Simulation
1. **gazebo_architecture.svg** - Gazebo system architecture (physics, rendering, ROS 2 bridge)
2. **sdf_model_structure.svg** - SDF model hierarchy (links, joints, plugins)
3. **gazebo_workflow.svg** - Simulation development workflow (design → model → test → deploy)
4. **sensor_simulation.svg** - How sensors are simulated in Gazebo
5. **physics_engine_comparison.svg** - Different physics engines (ODE, Bullet, Simbody, DART)

### Chapter 5: Unity Simulation
1. **unity_robotics_architecture.svg** - Unity + ROS 2 integration
2. **unity_vs_gazebo.svg** - Comparison of Unity and Gazebo strengths
3. **synthetic_data_pipeline.svg** - Using Unity for ML training data generation
4. **unity_simulation_flow.svg** - Scene setup and robot control workflow

## Topics Covered:

- Gazebo physics and rendering engines
- Robot model description (SDF/URDF)
- Sensor and actuator simulation
- ROS 2 integration patterns
- Unity for robotics applications
- Simulation-to-real transfer concepts

## Creation Notes:

- Diagrams should clearly show data flow and system boundaries
- Use consistent color coding (e.g., blue for Gazebo, orange for Unity, green for ROS 2)
- Include code snippets or configuration examples where helpful
- Maintain accessibility with alt text
- Export as SVG for scalability

## Status:

- Currently using ASCII diagrams embedded in chapter markdown
- TODO: Create professional SVG diagrams for production version
- ASCII diagrams serve as specifications for final graphics
