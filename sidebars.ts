import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Main tutorial sidebar - organized by chapters
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'intro',
      ],
    },
    {
      type: 'category',
      label: 'Part 1: Physical AI Fundamentals',
      items: [
        'chapter1_intro_physical_ai',
        'chapter2_embodied_intelligence',
      ],
    },
    {
      type: 'category',
      label: 'Part 2: ROS 2 Basics',
      items: [
        'chapter3_ros2_basics',
      ],
    },
    {
      type: 'category',
      label: 'Part 3: Robot Simulation',
      items: [
        'chapter4_gazebo_simulation',
        'chapter5_unity_simulation',
      ],
    },
    {
      type: 'category',
      label: 'Part 4: NVIDIA Isaac Sim',
      items: [
        'chapter6_intro_isaac_sim',
        'chapter7_isaac_perception_nav',
      ],
    },
    {
      type: 'category',
      label: 'Part 5: Vision-Language-Action Systems',
      items: [
        'chapter8_vla_systems',
        'chapter9_llm_voice_commands',
      ],
    },
    {
      type: 'category',
      label: 'Part 6: Perception & Vision',
      items: [
        'chapter10_computer_vision',
        'chapter11_3d_perception',
      ],
    },
  ],
};

export default sidebars;
