import React, { useEffect, useState, useRef } from 'react';
import styles from './styles.module.css';

interface StatItem {
  value: string;
  label: string;
  suffix?: string;
  icon: string;
}

interface AnimatedStatsProps {
  stats: StatItem[];
}

function AnimatedCounter({ value, suffix = '' }: { value: string; suffix?: string }) {
  const [count, setCount] = useState(0);
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);

  // Extract numeric value
  const numericValue = parseInt(value.replace(/\D/g, '')) || 0;

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !isVisible) {
          setIsVisible(true);
        }
      },
      { threshold: 0.5 }
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => observer.disconnect();
  }, [isVisible]);

  useEffect(() => {
    if (!isVisible) return;

    let startTime: number;
    const duration = 2000;

    const animate = (currentTime: number) => {
      if (!startTime) startTime = currentTime;
      const progress = Math.min((currentTime - startTime) / duration, 1);

      // Easing function for smooth animation
      const easeOutExpo = 1 - Math.pow(2, -10 * progress);
      setCount(Math.floor(easeOutExpo * numericValue));

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }, [isVisible, numericValue]);

  // Check if original value has + sign
  const hasPlus = value.includes('+');

  return (
    <span ref={ref} className={styles.counterValue}>
      {count}{hasPlus ? '+' : ''}{suffix}
    </span>
  );
}

export default function AnimatedStats({ stats }: AnimatedStatsProps): JSX.Element {
  return (
    <div className={styles.statsGrid}>
      {stats.map((stat, idx) => (
        <div
          key={idx}
          className={styles.statCard}
          style={{ animationDelay: `${idx * 0.1}s` }}
        >
          <div className={styles.statIcon}>{stat.icon}</div>
          <div className={styles.statContent}>
            <AnimatedCounter value={stat.value} suffix={stat.suffix} />
            <span className={styles.statLabel}>{stat.label}</span>
          </div>
          <div className={styles.cardGlow}></div>
          <div className={styles.borderGlow}></div>
        </div>
      ))}
    </div>
  );
}
