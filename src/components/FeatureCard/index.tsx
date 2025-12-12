import React from 'react';
import styles from './styles.module.css';
import Link from '@docusaurus/Link';

export interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  link?: string;
  linkText?: string;
  gradient?: 'purple' | 'blue' | 'green' | 'orange' | 'pink';
  delay?: number;
}

export default function FeatureCard({
  icon,
  title,
  description,
  link,
  linkText = 'Learn more',
  gradient = 'purple',
  delay = 0,
}: FeatureCardProps): JSX.Element {
  return (
    <div
      className={`${styles.card} ${styles[gradient]}`}
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className={styles.iconContainer}>
        <div className={styles.icon}>{icon}</div>
        <div className={styles.iconGlow}></div>
      </div>
      <h3 className={styles.title}>{title}</h3>
      <p className={styles.description}>{description}</p>
      {link && (
        <Link to={link} className={styles.link}>
          {linkText}
          <svg
            className={styles.arrow}
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
          >
            <path
              d="M6 12L10 8L6 4"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </Link>
      )}
    </div>
  );
}
