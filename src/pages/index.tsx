import React, { useState, useEffect, useRef } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import FeatureCard from '@site/src/components/FeatureCard';
import HeroRobot from '@site/src/components/HeroRobot';
import TypewriterText from '@site/src/components/TypewriterText';
import AnimatedStats from '@site/src/components/AnimatedStats';
import InteractiveBackground from '@site/src/components/InteractiveBackground';
import CircuitBoard from '@site/src/components/CircuitBoard';
import GlowingOrbs from '@site/src/components/GlowingOrbs';

import styles from './index.module.css';

// ============================================
// DATA CONFIGURATIONS
// ============================================

const features = [
  {
    icon: '🤖',
    title: 'Physical AI Fundamentals',
    description:
      'Master the foundations of embodied intelligence. Learn how robots perceive, reason, and interact with the physical world through sensors and actuators.',
    link: '/docs/chapter1_intro_physical_ai',
    linkText: 'Start learning',
    gradient: 'purple' as const,
  },
  {
    icon: '⚙️',
    title: 'ROS 2 Development',
    description:
      'Build robust robot applications with ROS 2. Explore nodes, topics, services, and actions while creating real-world robotic systems.',
    link: '/docs/chapter3_ros2_basics',
    linkText: 'Explore ROS 2',
    gradient: 'blue' as const,
  },
  {
    icon: '🎮',
    title: 'Robot Simulation',
    description:
      'Simulate robots in Gazebo, Unity, and NVIDIA Isaac Sim. Test algorithms safely before deploying to real hardware.',
    link: '/docs/chapter4_gazebo_simulation',
    linkText: 'Try simulations',
    gradient: 'green' as const,
  },
  {
    icon: '🧠',
    title: 'NVIDIA Isaac Platform',
    description:
      'Leverage NVIDIA Isaac for GPU-accelerated robotics. Train AI models and run real-time inference on robot hardware.',
    link: '/docs/chapter6_intro_isaac_sim',
    linkText: 'Discover Isaac',
    gradient: 'orange' as const,
  },
  {
    icon: '👁️',
    title: 'Vision-Language-Action',
    description:
      'Integrate cutting-edge VLA systems for intelligent robot control. Connect vision, language understanding, and robot actions.',
    link: '/docs/chapter8_vla_systems',
    linkText: 'Learn VLA',
    gradient: 'pink' as const,
  },
  {
    icon: '🚀',
    title: 'LLM & Voice Control',
    description:
      'Apply your knowledge with LLM integration and voice commands. Build intelligent conversational interfaces for robot control.',
    link: '/docs/chapter9_llm_voice_commands',
    linkText: 'Build projects',
    gradient: 'purple' as const,
  },
];

const stats = [
  { value: '11+', label: 'Chapters', icon: '📚' },
  { value: '50+', label: 'Code Examples', icon: '💻' },
  { value: '100', label: 'Free & Open', suffix: '%', icon: '🎁' },
  { value: '2025', label: 'Updated', icon: '✨' },
];

const typewriterTexts = [
  'Physical AI Systems',
  'Humanoid Robotics',
  'ROS 2 Development',
  'Computer Vision',
  'Neural Networks',
];

const techBadgesList = ['ROS 2', 'Python', 'CUDA', 'PyTorch', 'Isaac Sim'];

const technologies = [
  { name: 'ROS 2', icon: '🤖', color: '#6366f1' },
  { name: 'Gazebo', icon: '🌐', color: '#06b6d4' },
  { name: 'Unity', icon: '🎮', color: '#10b981' },
  { name: 'Isaac Sim', icon: '🎯', color: '#f59e0b' },
  { name: 'Python', icon: '🐍', color: '#3b82f6' },
  { name: 'C++', icon: '⚡', color: '#8b5cf6' },
  { name: 'PyTorch', icon: '🔥', color: '#ef4444' },
  { name: 'CUDA', icon: '💚', color: '#22c55e' },
];

// ============================================
// HERO SECTION
// ============================================

function HeroSection() {
  const [loaded, setLoaded] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0.5, y: 0.5 });
  const heroRef = useRef<HTMLElement>(null);

  useEffect(() => {
    // Trigger load animation
    const timer = setTimeout(() => setLoaded(true), 100);

    // Mouse tracking for spotlight effect
    const handleMouseMove = (e: MouseEvent) => {
      if (heroRef.current) {
        const rect = heroRef.current.getBoundingClientRect();
        setMousePosition({
          x: (e.clientX - rect.left) / rect.width,
          y: (e.clientY - rect.top) / rect.height,
        });
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => {
      clearTimeout(timer);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  return (
    <header ref={heroRef} className={styles.hero}>
      {/* ==========================================
          BACKGROUND LAYERS (z-index: 0-5)
          Ordered from back to front
          ========================================== */}

      {/* Layer 0: Solid base gradient */}
      <div className={styles.bgBase} />

      {/* Layer 1: Glowing orbs & aurora effect */}
      <div className={styles.bgOrbs}>
        <GlowingOrbs />
      </div>

      {/* Layer 2: Animated circuit board traces */}
      <div className={styles.bgCircuits}>
        <CircuitBoard />
      </div>

      {/* Layer 3: Interactive particle network */}
      <div className={styles.bgParticles}>
        <InteractiveBackground />
      </div>

      {/* Layer 4: Grid pattern overlay */}
      <div className={styles.bgGrid} />

      {/* Layer 5: Mouse-following spotlight */}
      <div
        className={styles.bgSpotlight}
        style={{
          background: `radial-gradient(800px circle at ${mousePosition.x * 100}% ${mousePosition.y * 100}%, rgba(99, 102, 241, 0.12), transparent 40%)`
        }}
      />

      {/* ==========================================
          OVERLAY EFFECTS (z-index: 6-8)
          ========================================== */}

      {/* Layer 6: Scan lines */}
      <div className={styles.overlayScanlines} />

      {/* Layer 7: Vignette */}
      <div className={styles.overlayVignette} />

      {/* Layer 8: Noise texture */}
      <div className={styles.overlayNoise} />

      {/* ==========================================
          MAIN CONTENT (z-index: 10)
          ========================================== */}

      <div className={clsx(styles.heroContent, loaded && styles.heroContentVisible)}>
        {/* Left Column: Text Content */}
        <div className={styles.heroTextColumn}>
          {/* Status Badge */}
          <div className={styles.heroBadge}>
            <span className={styles.heroBadgeDot} />
            <span className={styles.heroBadgeIcon}>🚀</span>
            <span className={styles.heroBadgeText}>Open Source Learning Platform</span>
            <span className={styles.heroBadgeTag}>NEW</span>
          </div>

          {/* Main Headline */}
          <Heading as="h1" className={styles.heroHeadline}>
            <span className={styles.heroHeadlineStatic}>Master</span>{' '}
            <span className={styles.heroHeadlineAnimated}>
              <TypewriterText
                texts={typewriterTexts}
                typingSpeed={80}
                deletingSpeed={40}
                pauseDuration={2500}
              />
            </span>
            <br />
            <span className={styles.heroHeadlineSub}>& Build the Future</span>
          </Heading>

          {/* Description */}
          <p className={styles.heroDescription}>
            Your comprehensive guide to building intelligent robots. Learn ROS 2,
            simulation, NVIDIA Isaac, computer vision, and Vision-Language-Action
            systems from the ground up with hands-on projects.
          </p>

          {/* CTA Buttons */}
          <div className={styles.heroActions}>
            <Link
              className={clsx(styles.btnPrimary)}
              to="/docs/intro"
            >
              <span className={styles.btnLabel}>Start Learning</span>
              <span className={styles.btnArrow}>→</span>
              <span className={styles.btnShine} />
            </Link>
            <Link
              className={clsx(styles.btnSecondary)}
              to="/docs/chapter3_ros2_basics"
            >
              <span className={styles.btnLabel}>Browse Topics</span>
              <svg className={styles.btnIconSvg} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </Link>
          </div>

          {/* Tech Stack Tags */}
          <div className={styles.heroTechTags}>
            {techBadgesList.map((tech, idx) => (
              <span
                key={tech}
                className={styles.heroTechTag}
                style={{ animationDelay: `${0.8 + idx * 0.1}s` }}
              >
                {tech}
              </span>
            ))}
          </div>

          {/* Stats */}
          <div className={styles.heroStats}>
            <AnimatedStats stats={stats} />
          </div>
        </div>

        {/* Right Column: Robot Visualization */}
        <div className={styles.heroVisualColumn}>
          <div className={styles.heroRobotContainer}>
            <HeroRobot />
            <div className={styles.heroRobotGlow} />
          </div>
        </div>
      </div>

      {/* ==========================================
          SCROLL INDICATOR (z-index: 10)
          ========================================== */}

      <div className={styles.scrollHint}>
        <div className={styles.scrollMouse}>
          <div className={styles.scrollWheel} />
        </div>
        <span className={styles.scrollLabel}>Scroll to explore</span>
        <div className={styles.scrollChevrons}>
          <span />
          <span />
          <span />
        </div>
      </div>

      {/* ==========================================
          BOTTOM FADE (z-index: 9)
          ========================================== */}

      <div className={styles.heroBottomFade} />
    </header>
  );
}

// ============================================
// FEATURES SECTION
// ============================================

function FeaturesSection() {
  return (
    <section className={styles.features}>
      <div className="container">
        <header className={styles.sectionHeader}>
          <span className={styles.sectionLabel}>What You'll Learn</span>
          <Heading as="h2" className={styles.sectionTitle}>
            Everything You Need to Build
            <br />
            <span className={styles.heroHeadlineAnimated}>Intelligent Robots</span>
          </Heading>
          <p className={styles.sectionDescription}>
            From fundamentals to advanced techniques, master the complete stack
            of modern robotics development.
          </p>
        </header>

        <div className={styles.featuresGrid}>
          {features.map((feature, idx) => (
            <FeatureCard
              key={idx}
              {...feature}
              delay={idx * 100}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

// ============================================
// TECH STACK SECTION
// ============================================

function TechStackSection() {
  return (
    <section className={styles.techStack}>
      <div className="container">
        <header className={styles.sectionHeader}>
          <span className={styles.sectionLabel}>Tech Stack</span>
          <Heading as="h2" className={styles.sectionTitle}>
            Industry-Standard Tools
          </Heading>
        </header>

        <div className={styles.techGrid}>
          {technologies.map((tech, idx) => (
            <div
              key={idx}
              className={styles.techCard}
              style={{
                animationDelay: `${idx * 0.05}s`,
                '--tech-accent': tech.color
              } as React.CSSProperties}
            >
              <span className={styles.techCardIcon}>{tech.icon}</span>
              <span className={styles.techCardName}>{tech.name}</span>
              <div className={styles.techCardGlow} />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ============================================
// CTA SECTION
// ============================================

function CTASection() {
  return (
    <section className={styles.cta}>
      <div className="container">
        <div className={styles.ctaCard}>
          {/* Background decoration */}
          <div className={styles.ctaBg}>
            <div className={styles.ctaOrb1} />
            <div className={styles.ctaOrb2} />
            <div className={styles.ctaOrb3} />
            <div className={styles.ctaGridPattern} />
          </div>

          {/* Content */}
          <div className={styles.ctaContent}>
            <Heading as="h2" className={styles.ctaHeadline}>
              Ready to Build the Future?
            </Heading>
            <p className={styles.ctaText}>
              Start your journey into Physical AI and robotics today. Our
              comprehensive curriculum will take you from beginner to expert.
            </p>
            <Link
              className={styles.ctaBtn}
              to="/docs/intro"
            >
              Begin Your Journey
              <span className={styles.ctaBtnArrow}>→</span>
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

// ============================================
// MAIN PAGE COMPONENT
// ============================================

export default function Home(): JSX.Element {
  return (
    <Layout
      title="Learn Physical AI & Robotics"
      description="Comprehensive guide to Physical AI, ROS 2, robot simulation, and Vision-Language-Action systems for humanoid robotics"
    >
      <HeroSection />
      <main>
        <FeaturesSection />
        <TechStackSection />
        <CTASection />
      </main>
    </Layout>
  );
}
