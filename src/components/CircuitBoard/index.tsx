import React, { useEffect, useRef } from 'react';
import styles from './styles.module.css';

export default function CircuitBoard(): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    interface CircuitPath {
      points: { x: number; y: number }[];
      color: string;
      progress: number;
      speed: number;
      width: number;
    }

    const circuits: CircuitPath[] = [];
    const colors = ['#6366f1', '#06b6d4', '#8b5cf6', '#10b981'];

    // Generate circuit paths
    const generateCircuit = (): CircuitPath => {
      const points: { x: number; y: number }[] = [];
      const startX = Math.random() * canvas.width;
      const startY = Math.random() * canvas.height;
      let x = startX;
      let y = startY;

      points.push({ x, y });

      const segments = Math.floor(Math.random() * 8) + 4;
      for (let i = 0; i < segments; i++) {
        const direction = Math.floor(Math.random() * 4);
        const length = Math.random() * 100 + 50;

        switch (direction) {
          case 0: x += length; break;
          case 1: x -= length; break;
          case 2: y += length; break;
          case 3: y -= length; break;
        }

        // Add a node point
        points.push({ x, y });

        // Sometimes branch at 90 degrees
        if (Math.random() > 0.7) {
          const branchDir = direction % 2 === 0 ? (Math.random() > 0.5 ? 2 : 3) : (Math.random() > 0.5 ? 0 : 1);
          const branchLength = Math.random() * 50 + 20;

          switch (branchDir) {
            case 0: x += branchLength; break;
            case 1: x -= branchLength; break;
            case 2: y += branchLength; break;
            case 3: y -= branchLength; break;
          }
          points.push({ x, y });
        }
      }

      return {
        points,
        color: colors[Math.floor(Math.random() * colors.length)],
        progress: 0,
        speed: Math.random() * 0.01 + 0.005,
        width: Math.random() * 2 + 1,
      };
    };

    // Initialize circuits
    for (let i = 0; i < 15; i++) {
      circuits.push(generateCircuit());
    }

    const drawNode = (x: number, y: number, color: string, size: number) => {
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Inner glow
      ctx.beginPath();
      ctx.arc(x, y, size * 0.5, 0, Math.PI * 2);
      ctx.fillStyle = 'white';
      ctx.fill();
    };

    const animate = () => {
      ctx.fillStyle = 'rgba(10, 10, 26, 0.03)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      circuits.forEach((circuit, index) => {
        const { points, color, progress, width } = circuit;

        if (points.length < 2) return;

        // Calculate total path length
        let totalLength = 0;
        for (let i = 1; i < points.length; i++) {
          const dx = points[i].x - points[i - 1].x;
          const dy = points[i].y - points[i - 1].y;
          totalLength += Math.sqrt(dx * dx + dy * dy);
        }

        // Draw the path up to current progress
        const targetLength = totalLength * progress;
        let currentLength = 0;

        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);

        for (let i = 1; i < points.length; i++) {
          const dx = points[i].x - points[i - 1].x;
          const dy = points[i].y - points[i - 1].y;
          const segmentLength = Math.sqrt(dx * dx + dy * dy);

          if (currentLength + segmentLength <= targetLength) {
            ctx.lineTo(points[i].x, points[i].y);
            currentLength += segmentLength;
          } else {
            const remainingLength = targetLength - currentLength;
            const ratio = remainingLength / segmentLength;
            const x = points[i - 1].x + dx * ratio;
            const y = points[i - 1].y + dy * ratio;
            ctx.lineTo(x, y);

            // Draw leading glow
            drawNode(x, y, color, 4);
            break;
          }
        }

        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.shadowColor = color;
        ctx.shadowBlur = 8;
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Draw nodes at junction points
        for (let i = 0; i < points.length; i++) {
          const dx = i > 0 ? points[i].x - points[i - 1].x : 0;
          const dy = i > 0 ? points[i].y - points[i - 1].y : 0;
          const pointLength = i > 0 ? Math.sqrt(dx * dx + dy * dy) : 0;
          let accumulatedLength = 0;

          for (let j = 1; j <= i; j++) {
            const pdx = points[j].x - points[j - 1].x;
            const pdy = points[j].y - points[j - 1].y;
            accumulatedLength += Math.sqrt(pdx * pdx + pdy * pdy);
          }

          if (accumulatedLength <= targetLength) {
            drawNode(points[i].x, points[i].y, color, 3);
          }
        }

        // Update progress
        circuit.progress += circuit.speed;

        // Reset when complete
        if (circuit.progress >= 1) {
          circuits[index] = generateCircuit();
        }
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return <canvas ref={canvasRef} className={styles.circuit} />;
}
