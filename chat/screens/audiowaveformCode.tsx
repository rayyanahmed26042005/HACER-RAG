import React, { useEffect } from 'react';
import { View, Animated, Easing, StyleSheet } from 'react-native';

const WaveformAnimation = () => {
  const waveAnim = new Animated.Value(0);

  useEffect(() => {
    const startWaveAnimation = () => {
      Animated.loop(
        Animated.sequence([
          Animated.timing(waveAnim, {
            toValue: 1,
            duration: 500,
            easing: Easing.linear,
            useNativeDriver: true,
          }),
          Animated.timing(waveAnim, {
            toValue: 0,
            duration: 500,
            easing: Easing.linear,
            useNativeDriver: true,
          }),
        ])
      ).start();
    };

    startWaveAnimation();

    return () => {
      waveAnim.stopAnimation();
    };
  }, [waveAnim]);

  const waveStyle = {
    opacity: waveAnim.interpolate({
      inputRange: [0, 1],
      outputRange: [0.5, 1],
    }),
    transform: [
      {
        scale: waveAnim.interpolate({
          inputRange: [0, 1],
          outputRange: [1, 1.3],
        }),
      },
    ],
  };

  return (
    <View style={styles.container}>
      <Animated.View style={[styles.waveform, waveStyle]} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
    height: 100,
    marginBottom: 10,
  },
  waveform: {
    width: 100,
    height: 100,
    backgroundColor: '#0078fe',
    borderRadius: 50,
  },
});

export default WaveformAnimation;
