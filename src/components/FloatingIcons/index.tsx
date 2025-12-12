import React, { useState } from 'react';
import styles from './styles.module.css';

const techIcons = [
  { icon: '🤖', label: 'ROS 2', color: '#6366f1' },
  { icon: '🧠', label: 'Neural Networks', color: '#ec4899' },
  { icon: '👁️', label: 'Computer Vision', color: '#06b6d4' },
  { icon: '🎮', label: 'Simulation', color: '#10b981' },
  { icon: '⚡', label: 'CUDA', color: '#f59e0b' },
  { icon: '🔧', label: 'Isaac Sim', color: '#8b5cf6' },
  { icon: '🎯', label: 'Motion Control', color: '#ef4444' },
  { icon: '🌐', label: 'Gazebo', color: '#3b82f6' },
];

export default function FloatingIcons(): JSX.Element {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  return (
    <div className={styles.floatingContainer}>
      {techIcons.map((tech, index) => (
        <div
          key={index}
          className={styles.iconWrapper}
          style={{
            '--delay': `${index * 0.5}s`,
            '--angle': `${(index * 360) / techIcons.length}deg`,
            '--color': tech.color,
          } as React.CSSProperties}
          onMouseEnter={() => setHoveredIndex(index)}
          onMouseLeave={() => setHoveredIndex(null)}
        >
          <div className={`${styles.iconBubble} ${hoveredIndex === index ? styles.hovered : ''}`}>
            <span className={styles.icon}>{tech.icon}</span>
            <div className={styles.tooltip}>
              <span className={styles.tooltipText}>{tech.label}</span>
              <div className={styles.tooltipArrow}></div>
            </div>
            <div className={styles.glowRing}></div>
          </div>
          <div className={styles.connectionLine}></div>
        </div>
      ))}
      <div className={styles.centerOrb}>
        <div className={styles.orbCore}></div>
        <div className={styles.orbRing1}></div>
        <div className={styles.orbRing2}></div>
      </div>
    </div>
  );
}
