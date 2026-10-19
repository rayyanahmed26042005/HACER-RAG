import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  TouchableOpacity,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import Swiper from 'react-native-swiper';
import { useProfile } from '../utils/ProfileContext';

type RootStackParamList = {
  Chat: undefined;
  ProfileSelect: undefined;
};

const Intro = () => {
  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const { profileId } = useProfile();

  return (
    <View style={styles.mainContainer}>
      <Swiper
        loop={false}
        showsPagination={true}
        dotColor="#999"
        activeDotColor="#fbd85d"
      >
        {/* Screen 1 */}
        <View style={styles.container}>
          <Image
            source={require('./llm.png')}
            style={styles.image}
            resizeMode="contain"
            onError={(e) => console.log('Image error llm.png:', e.nativeEvent.error)}
          />
          <View style={styles.textContainer}>
            <Text style={styles.title}>
              Meet <Text style={styles.highlight}>Genix</Text>!
            </Text>
            <Text style={styles.subtitle}>Your own AI assistant</Text>
            <Text style={styles.description}>
              Ask your questions and receive articles using{'\n'}
              artificial intelligence assistant
            </Text>
          </View>
        </View>

        {/* Screen 2 */}
        <View style={styles.container}>
          <Image
            source={require('./tasks.png')}
            style={styles.image}
            resizeMode="contain"
            onError={(e) => console.log('Image error tasks.png:', e.nativeEvent.error)}
          />
          <Text style={styles.title}>Welcome to Our App!</Text>
          <Text style={styles.description}>We're glad to have you on board.</Text>
          <TouchableOpacity
            style={styles.button}
            onPress={() => {
              if (profileId) {
                navigation.replace('Chat');
              } else {
                navigation.replace('ProfileSelect');
              }
            }}
          >
            <Text style={styles.buttonText}>Get Started</Text>
          </TouchableOpacity>
        </View>
      </Swiper>
    </View>
  );
};

const styles = StyleSheet.create({
  mainContainer: {
    flex: 1,
    backgroundColor: '#000112',
  },
  container: {
    flex: 1,
    backgroundColor: '#000112',
    paddingHorizontal: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  image: {
    width: 300,
    height: 350,
    marginTop: 70,
  },
  textContainer: {
    alignItems: 'center',
  },
  title: {
    fontSize: 26,
    color: '#fff',
    fontWeight: '600',
  },
  highlight: {
    color: '#fbd85d',
  },
  subtitle: {
    fontSize: 20,
    color: '#fff',
    marginTop: 10,
  },
  description: {
    fontSize: 14,
    color: '#ccc',
    textAlign: 'center',
    marginTop: 8,
    marginBottom: 24,
  },
  button: {
    backgroundColor: '#a06ef4',
    paddingVertical: 14,
    paddingHorizontal: 50,
    borderRadius: 30,
    marginTop: 20,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default Intro;