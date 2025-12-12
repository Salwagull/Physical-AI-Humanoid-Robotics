import React, { useRef, useState } from 'react';
import styles from './styles.module.css';

interface MagneticButtonProps {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  href?: string;
  strength?: number;
}

export default function MagneticButton({
  children,
  className = '',
  onClick,
  href,
  strength = 0.3,
}: MagneticButtonProps): JSX.Element {
  const buttonRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!buttonRef.current) return;

    const rect = buttonRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    const distanceX = e.clientX - centerX;
    const distanceY = e.clientY - centerY;

    setPosition({
      x: distanceX * strength,
      y: distanceY * strength,
    });
  };

  const handleMouseLeave = () => {
    setPosition({ x: 0, y: 0 });
    setIsHovered(false);
  };

  const handleMouseEnter = () => {
    setIsHovered(true);
  };

  const content = (
    <div
      ref={buttonRef}
      className={`${styles.magneticWrapper} ${className}`}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onMouseEnter={handleMouseEnter}
      onClick={onClick}
      style={{
        transform: `translate(${position.x}px, ${position.y}px)`,
      }}
    >
      <div
        className={styles.magneticContent}
        style={{
          transform: `translate(${position.x * 0.5}px, ${position.y * 0.5}px)`,
        }}
      >
        {children}
      </div>
      <div className={`${styles.magneticGlow} ${isHovered ? styles.active : ''}`} />
      <div className={`${styles.magneticRipple} ${isHovered ? styles.active : ''}`} />
    </div>
  );

  if (href) {
    return <a href={href} className={styles.magneticLink}>{content}</a>;
  }

  return content;
}
