// chat/App.tsx - full updated file

import React, { useEffect, useState } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { ActivityIndicator, Text, View } from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';

// Screens
import Intro from './screens/intro';
import ChatScreen from './screens/ChatScreen';
import Models from './screens/Models';
import Settings from './screens/Settings';
import ProfileSelectScreen from './screens/ProfileSelectScreen';
import DocumentsScreen from './screens/DocumentsScreen';


import { SafeAreaProvider } from 'react-native-safe-area-context';
import { ProfileProvider, useProfile } from './utils/ProfileContext';

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

const AppContent = () => {
  const [isFirstLaunch, setIsFirstLaunch] = useState<boolean | null>(null);

  useEffect(() => {
    const checkFirstLaunch = async () => {
      try {
        const hasLaunched = await AsyncStorage.getItem('hasLaunched');
        if (hasLaunched === null) {
          await AsyncStorage.setItem('hasLaunched', 'true');
          setIsFirstLaunch(true);
        } else {
          setIsFirstLaunch(false);
        }
      } catch (error) {
        console.error('Error checking first launch:', error);
        setIsFirstLaunch(false);
      }
    };

    checkFirstLaunch();
  }, []);

  const { profileId, isLoading: profileLoading } = useProfile();

  if (isFirstLaunch === null || profileLoading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#0B030F' }}>
        <ActivityIndicator size="large" color="#fbd85d" />
        <Text style={{ color: '#fbd85d', marginTop: 10 }}>Loading...</Text>
      </View>
    );
  }

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {isFirstLaunch && (
          <Stack.Screen name="Intro" component={Intro} />
        )}
        {!profileId ? (
          <Stack.Screen name="ProfileSelect" component={ProfileSelectScreen} />
        ) : (
          <>
            <Stack.Screen name="Chat" component={ChatScreen} />
            <Stack.Screen name="Models" component={Models} />
            <Stack.Screen name="Documents" component={DocumentsScreen} />
            <Stack.Screen name="Settings" component={Settings} />
          </>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default () => (
  <SafeAreaProvider>
    <ProfileProvider>
      <AppContent />
    </ProfileProvider>
  </SafeAreaProvider>
);
