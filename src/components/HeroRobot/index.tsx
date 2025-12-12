import React, { useEffect, useRef, useState } from 'react';
import styles from './styles.module.css';

export default function HeroRobot(): JSX.Element {
  const robotRef = useRef<HTMLDivElement>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (robotRef.current) {
        const rect = robotRef.current.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const rotateX = (e.clientY - centerY) / 50;
        const rotateY = (e.clientX - centerX) / 50;
        setMousePos({ x: rotateY, y: -rotateX });
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className={styles.robotContainer} ref={robotRef}>
      {/* 3D Holographic Ring */}
      <div className={styles.holoRing}>
        <div className={styles.ringOuter}></div>
        <div className={styles.ringMiddle}></div>
        <div className={styles.ringInner}></div>
      </div>

      {/* Main Robot Body with 3D transform */}
      <div
        className={styles.robotBody}
        style={{
          transform: `perspective(1000px) rotateX(${mousePos.y}deg) rotateY(${mousePos.x}deg)`
        }}
      >
        {/* Robot Head */}
        <div className={styles.robotHead}>
          <div className={styles.visor}>
            <div className={styles.eyeLeft}>
              <div className={styles.eyePupil}></div>
            </div>
            <div className={styles.eyeRight}>
              <div className={styles.eyePupil}></div>
            </div>
            <div className={styles.scanLine}></div>
          </div>
          <div className={styles.antenna}>
            <div className={styles.antennaOrb}></div>
          </div>
        </div>

        {/* Robot Torso */}
        <div className={styles.robotTorso}>
          <div className={styles.chestPlate}>
            <div className={styles.arcReactor}>
              <div className={styles.reactorCore}></div>
              <div className={styles.reactorRing}></div>
              <div className={styles.reactorRing2}></div>
            </div>
          </div>
          <div className={styles.circuitLines}>
            {[...Array(6)].map((_, i) => (
              <div key={i} className={styles.circuit} style={{ animationDelay: `${i * 0.2}s` }}></div>
            ))}
          </div>
        </div>

        {/* Robot Arms */}
        <div className={styles.armLeft}>
          <div className={styles.armSegment}></div>
          <div className={styles.armJoint}></div>
          <div className={styles.forearm}>
            <div className={styles.handGlow}></div>
          </div>
        </div>
        <div className={styles.armRight}>
          <div className={styles.armSegment}></div>
          <div className={styles.armJoint}></div>
          <div className={styles.forearm}>
            <div className={styles.handGlow}></div>
          </div>
        </div>
      </div>

      {/* Floating Data Points */}
      <div className={styles.dataPoints}>
        {[...Array(8)].map((_, i) => (
          <div
            key={i}
            className={styles.dataPoint}
            style={{
              animationDelay: `${i * 0.3}s`,
              left: `${10 + (i % 4) * 25}%`,
              top: `${20 + Math.floor(i / 4) * 50}%`
            }}
          >
            <span className={styles.dataValue}>{['ROS2', 'CUDA', 'AI', 'ML', 'CV', 'NLP', 'SIM', 'VLA'][i]}</span>
          </div>
        ))}
      </div>

      {/* Energy Pulse Effect */}
      <div className={styles.energyPulse}></div>
      <div className={styles.energyPulse2}></div>
    </div>
  );
}
