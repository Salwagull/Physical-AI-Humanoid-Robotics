import React, { useEffect, useRef } from 'react';
import styles from './styles.module.css';

export default function CodeRain(): JSX.Element {
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

    // Robotics and AI related characters
    const chars = 'ROSAI01CUDA<>{}[]()=>∫∑∏πθλμσ∂∇⊗⊕⊖∀∃∈∉⊂⊃∪∩αβγδεηικνρτφψωΣΩ';
    const charArray = chars.split('');

    const fontSize = 14;
    const columns = Math.floor(canvas.width / fontSize);
    const drops: number[] = new Array(columns).fill(1);
    const speeds: number[] = new Array(columns).fill(0).map(() => Math.random() * 0.5 + 0.3);
    const colors = ['#6366f1', '#8b5cf6', '#06b6d4', '#10b981'];

    const draw = () => {
      // Fade effect
      ctx.fillStyle = 'rgba(10, 10, 26, 0.05)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.font = `${fontSize}px 'JetBrains Mono', monospace`;

      for (let i = 0; i < drops.length; i++) {
        const char = charArray[Math.floor(Math.random() * charArray.length)];
        const x = i * fontSize;
        const y = drops[i] * fontSize;

        // Color based on position
        const colorIndex = Math.floor(i / (columns / colors.length)) % colors.length;
        const baseColor = colors[colorIndex];

        // Leading character is brighter
        ctx.fillStyle = baseColor;
        ctx.shadowColor = baseColor;
        ctx.shadowBlur = 10;
        ctx.fillText(char, x, y);

        // Trail characters are dimmer
        if (Math.random() > 0.98) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.shadowBlur = 20;
          ctx.fillText(char, x, y);
        }

        ctx.shadowBlur = 0;

        // Reset drop when it goes off screen
        if (y > canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
        }

        drops[i] += speeds[i];
      }

      animationRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return <canvas ref={canvasRef} className={styles.codeRain} />;
}
