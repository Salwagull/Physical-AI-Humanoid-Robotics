import React, { useEffect, useRef, useState } from 'react';
import styles from './styles.module.css';

interface Shape {
  id: number;
  type: 'circle' | 'square' | 'triangle' | 'hexagon' | 'ring';
  x: number;
  y: number;
  size: number;
  rotation: number;
  speed: number;
  depth: number;
  color: string;
}

export default function ParallaxShapes(): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [shapes] = useState<Shape[]>(() => {
    const colors = ['#6366f1', '#8b5cf6', '#06b6d4', '#ec4899', '#10b981'];
    const types: Shape['type'][] = ['circle', 'square', 'triangle', 'hexagon', 'ring'];

    return Array.from({ length: 20 }, (_, i) => ({
      id: i,
      type: types[Math.floor(Math.random() * types.length)],
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 60 + 20,
      rotation: Math.random() * 360,
      speed: Math.random() * 0.5 + 0.2,
      depth: Math.random() * 3 + 1,
      color: colors[Math.floor(Math.random() * colors.length)],
    }));
  });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const x = (e.clientX - rect.left - rect.width / 2) / rect.width;
        const y = (e.clientY - rect.top - rect.height / 2) / rect.height;
        setMousePos({ x, y });
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const renderShape = (shape: Shape) => {
    const parallaxX = mousePos.x * 50 * shape.depth;
    const parallaxY = mousePos.y * 50 * shape.depth;

    const style: React.CSSProperties = {
      left: `${shape.x}%`,
      top: `${shape.y}%`,
      width: shape.size,
      height: shape.size,
      transform: `translate(${parallaxX}px, ${parallaxY}px) rotate(${shape.rotation}deg)`,
      '--shape-color': shape.color,
      '--animation-duration': `${20 / shape.speed}s`,
      opacity: 0.15 + (shape.depth * 0.1),
    } as React.CSSProperties;

    return (
      <div
        key={shape.id}
        className={`${styles.shape} ${styles[shape.type]}`}
        style={style}
      />
    );
  };

  return (
    <div ref={containerRef} className={styles.container}>
      {shapes.map(renderShape)}
    </div>
  );
}
