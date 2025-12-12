import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/Physical-AI-Humanoid-Robotics/docs',
    component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs', 'c71'),
    routes: [
      {
        path: '/Physical-AI-Humanoid-Robotics/docs',
        component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs', 'cb5'),
        routes: [
          {
            path: '/Physical-AI-Humanoid-Robotics/docs',
            component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs', 'f46'),
            routes: [
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/assets/isaac_sim_diagrams',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/assets/isaac_sim_diagrams', '8af'),
                exact: true
              },
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/assets/physical_ai_diagrams',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/assets/physical_ai_diagrams', '34c'),
                exact: true
              },
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/assets/ros2_diagrams',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/assets/ros2_diagrams', 'b3b'),
                exact: true
              },
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/assets/simulation_diagrams',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/assets/simulation_diagrams', 'bb4'),
                exact: true
              },
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/assets/vla_diagrams',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/assets/vla_diagrams', '62d'),
                exact: true
              },
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/chapter1_intro_physical_ai',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/chapter1_intro_physical_ai', 'e6e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/chapter10_computer_vision',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/chapter10_computer_vision', 'dd4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/chapter11_3d_perception',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/chapter11_3d_perception', 'b50'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/chapter2_embodied_intelligence',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/chapter2_embodied_intelligence', '765'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/chapter3_ros2_basics',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/chapter3_ros2_basics', 'be3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/chapter4_gazebo_simulation',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/chapter4_gazebo_simulation', '3db'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/chapter5_unity_simulation',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/chapter5_unity_simulation', 'f3a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/chapter6_intro_isaac_sim',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/chapter6_intro_isaac_sim', 'fbf'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/chapter7_isaac_perception_nav',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/chapter7_isaac_perception_nav', '8f3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/chapter8_vla_systems',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/chapter8_vla_systems', 'eae'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/chapter9_llm_voice_commands',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/chapter9_llm_voice_commands', 'd6c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Physical-AI-Humanoid-Robotics/docs/intro',
                component: ComponentCreator('/Physical-AI-Humanoid-Robotics/docs/intro', '60b'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/Physical-AI-Humanoid-Robotics/',
    component: ComponentCreator('/Physical-AI-Humanoid-Robotics/', '6c1'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
