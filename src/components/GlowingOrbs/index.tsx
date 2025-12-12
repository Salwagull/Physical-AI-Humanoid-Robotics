import React from 'react';
import styles from './styles.module.css';

export default function GlowingOrbs(): JSX.Element {
  return (
    <div className={styles.orbContainer}>
      {/* Large primary orbs */}
      <div className={`${styles.orb} ${styles.orb1}`}>
        <div className={styles.orbInner}></div>
      </div>
      <div className={`${styles.orb} ${styles.orb2}`}>
        <div className={styles.orbInner}></div>
      </div>
      <div className={`${styles.orb} ${styles.orb3}`}>
        <div className={styles.orbInner}></div>
      </div>
      <div className={`${styles.orb} ${styles.orb4}`}>
        <div className={styles.orbInner}></div>
      </div>

      {/* Ambient light spots */}
      <div className={`${styles.lightSpot} ${styles.light1}`}></div>
      <div className={`${styles.lightSpot} ${styles.light2}`}></div>
      <div className={`${styles.lightSpot} ${styles.light3}`}></div>

      {/* Gradient mesh */}
      <div className={styles.gradientMesh}></div>

      {/* Aurora effect */}
      <div className={styles.aurora}>
        <div className={styles.auroraLayer1}></div>
        <div className={styles.auroraLayer2}></div>
        <div className={styles.auroraLayer3}></div>
      </div>
    </div>
  );
}
