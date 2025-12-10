import React, { useRef, useState } from 'react';
import { View, Button, Image } from 'react-native';
import { Camera, useCameraDevices } from 'react-native-vision-camera';

export default function cameraCapture(props) {
  const [photoUri, setPhotoUri] = useState(null);
  const camera = useRef(null);
  const devices = useCameraDevices();
  const device = devices.back;

  const takePhoto = async () => {
    if (!device || !camera.current) return;
    const photo = await camera.current.takePhoto({ quality: 85 });
    setPhotoUri('file://' + photo.path);

    // Tell Hyperview we have a photo
    props.onUpdate?.({ action: 'photoTaken', uri: photoUri });
  };

  return (
    <View style={{ flex: 1 }}>
      {!photoUri ? (
        device && <Camera style={{ flex: 1 }} ref={camera} device={device} isActive={true} photo={true} />
      ) : (
        <Image source={{ uri: photoUri }} style={{ flex: 1 }} />
      )}
      <Button title="Take Photo" onPress={takePhoto} />
    </View>
  );
}

cameraCapture.namespaceURI = 'https://myapp.com/hyperview-camera';
cameraCapture.localName = 'camera-capture';
