import React from 'react';
import { SafeAreaView } from 'react-native';
import Hyperview from 'hyperview';
import cameraCapture from './cameraCapture';

export default function App() {
  return (
    <SafeAreaView style={{ flex: 1 }}>
      <Hyperview
        entrypointUrl="https://example.com/screen/camera" // to be determined HXML endpoint
        components={[cameraCapture]}
        fetch={fetch}
      />
    </SafeAreaView>
  );
}
