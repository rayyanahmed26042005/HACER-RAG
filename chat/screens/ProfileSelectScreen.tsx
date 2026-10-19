import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  FlatList,
  Modal,
  TextInput,
  Alert,
  ActivityIndicator,
} from 'react-native';
import {
  Profile,
  getProfiles,
  createProfile,
  deleteProfile,
  verifyPin,
} from '../utils/profileManager';
import { useProfile } from '../utils/ProfileContext';
import Icon from 'react-native-vector-icons/MaterialIcons';

const ProfileSelectScreen = () => {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Create Profile Modal
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState('');
  const [newPin, setNewPin] = useState('');
  const [newPinEnabled, setNewPinEnabled] = useState(false);
  const [creating, setCreating] = useState(false);

  // Pin Unlock Modal
  const [showPinUnlock, setShowPinUnlock] = useState(false);
  const [unlockingProfile, setUnlockingProfile] = useState<Profile | null>(null);
  const [enteredPin, setEnteredPin] = useState('');

  const { switchProfile } = useProfile();

  useEffect(() => {
    loadProfiles();
  }, []);

  const loadProfiles = async () => {
    try {
      const p = await getProfiles();
      setProfiles(p);
    } catch (e) {
      console.error('Error loading profiles', e);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateProfile = async () => {
    if (!newName.trim()) {
      Alert.alert('Error', 'Profile name is required');
      return;
    }

    if (newPinEnabled && newPin.length !== 4) {
      Alert.alert('Error', 'PIN must be exactly 4 digits');
      return;
    }

    if (creating) return; // Bug #13 fix: prevent double-tap
    setCreating(true);

    try {
      const p = await createProfile(newName, newPinEnabled ? newPin : undefined);
      setProfiles([...profiles, p]);
      setShowCreate(false);
      setNewName('');
      setNewPin('');
      setNewPinEnabled(false);
    } catch (e) {
      Alert.alert('Error', 'Failed to create profile');
      console.error(e);
    } finally {
      setCreating(false);
    }
  };

  const handleSelectProfile = (profile: Profile) => {
    if (profile.pinEnabled) {
      setUnlockingProfile(profile);
      setEnteredPin('');
      setShowPinUnlock(true);
    } else {
      enterProfile(profile.id);
    }
  };

  const verifyAndEnter = async () => {
    if (!unlockingProfile) return;
    
    const isValid = await verifyPin(unlockingProfile.id, enteredPin);
    if (!isValid) {
      Alert.alert('Error', 'Incorrect PIN');
      setEnteredPin('');
      return;
    }
    
    setShowPinUnlock(false);
    enterProfile(unlockingProfile.id);
  };

  const enterProfile = async (id: string) => {
    await switchProfile(id);
  };

  const handleDeleteProfile = (profile: Profile) => {
    Alert.alert(
      'Delete Profile',
      `Are you sure you want to delete ${profile.name}? All chats, documents, and settings will be permanently lost.`,
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Delete', 
          style: 'destructive',
          onPress: async () => {
            await deleteProfile(profile.id);
            setProfiles(profiles.filter(p => p.id !== profile.id));
          }
        }
      ]
    );
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#8B5CF6" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Who's chatting?</Text>
      
      <FlatList
        data={profiles}
        keyExtractor={item => item.id}
        numColumns={2}
        columnWrapperStyle={styles.row}
        contentContainerStyle={styles.listContainer}
        renderItem={({ item }) => (
          <TouchableOpacity
            style={styles.profileCard}
            onPress={() => handleSelectProfile(item)}
            onLongPress={() => handleDeleteProfile(item)}
            delayLongPress={800}
          >
            <View style={[styles.avatarCircle, { backgroundColor: item.avatarColor }]}>
              <Text style={styles.avatarText}>
                {item.name.charAt(0).toUpperCase()}
              </Text>
            </View>
            <Text style={styles.profileName} numberOfLines={1}>{item.name}</Text>
            {item.pinEnabled && (
              <Icon name="lock" size={14} color="#9CA3AF" style={styles.lockIcon} />
            )}
          </TouchableOpacity>
        )}
        ListFooterComponent={
          <TouchableOpacity style={styles.createCard} onPress={() => setShowCreate(true)}>
            <View style={styles.createCircle}>
              <Icon name="add" size={32} color="#F9FAFB" />
            </View>
            <Text style={styles.createText}>Add Profile</Text>
          </TouchableOpacity>
        }
      />

      {/* CREATE PROFILE MODAL */}
      <Modal visible={showCreate} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>New Profile</Text>
            
            <TextInput
              style={styles.input}
              placeholder="Profile Name"
              placeholderTextColor="#9CA3AF"
              value={newName}
              onChangeText={setNewName}
              autoFocus
            />

            <TouchableOpacity 
              style={styles.toggleRow} 
              onPress={() => setNewPinEnabled(!newPinEnabled)}
            >
              <Text style={styles.toggleText}>Enable PIN protection</Text>
              <View style={[styles.checkbox, newPinEnabled && styles.checkboxActive]}>
                {newPinEnabled && <Icon name="check" size={16} color="#fff" />}
              </View>
            </TouchableOpacity>

            {newPinEnabled && (
              <TextInput
                style={styles.input}
                placeholder="4-Digit PIN"
                placeholderTextColor="#9CA3AF"
                value={newPin}
                onChangeText={text => setNewPin(text.replace(/[^0-9]/g, '').slice(0,4))}
                keyboardType="numeric"
                secureTextEntry
              />
            )}

            <View style={styles.modalActions}>
              <TouchableOpacity style={styles.cancelBtn} onPress={() => setShowCreate(false)}>
                <Text style={styles.cancelBtnText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.saveBtn, creating && { opacity: 0.5 }]} onPress={handleCreateProfile} disabled={creating}>
                <Text style={styles.saveBtnText}>{creating ? 'Creating...' : 'Create'}</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* PIN UNLOCK MODAL */}
      <Modal visible={showPinUnlock} transparent animationType="fade">
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Enter PIN</Text>
            <Text style={styles.modalSubtitle}>Unlock {unlockingProfile?.name}</Text>
            
            <TextInput
              style={[styles.input, styles.pinInput]}
              value={enteredPin}
              onChangeText={text => setEnteredPin(text.replace(/[^0-9]/g, '').slice(0,4))}
              keyboardType="numeric"
              secureTextEntry
              autoFocus
            />

            <View style={styles.modalActions}>
              <TouchableOpacity style={styles.cancelBtn} onPress={() => {
                setShowPinUnlock(false);
                setEnteredPin('');
              }}>
                <Text style={styles.cancelBtnText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                style={[styles.saveBtn, enteredPin.length !== 4 && { opacity: 0.5 }]} 
                onPress={verifyAndEnter}
                disabled={enteredPin.length !== 4}
              >
                <Text style={styles.saveBtnText}>Unlock</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1, backgroundColor: '#0B0F19', justifyContent: 'center', alignItems: 'center'
  },
  container: {
    flex: 1, backgroundColor: '#0B0F19', paddingTop: 60,
  },
  title: {
    fontSize: 28, color: '#F9FAFB', fontWeight: 'bold', textAlign: 'center', marginBottom: 40, fontFamily: 'sans-serif'
  },
  listContainer: {
    paddingHorizontal: 20, paddingBottom: 40,
  },
  row: {
    justifyContent: 'flex-start', flexWrap: 'wrap',
  },
  profileCard: {
    width: '45%', alignItems: 'center', margin: '2.5%', marginBottom: 24,
  },
  avatarCircle: {
    width: 100, height: 100, borderRadius: 50, justifyContent: 'center', alignItems: 'center', marginBottom: 12,
  },
  avatarText: {
    fontSize: 40, color: '#111827', fontWeight: 'bold', fontFamily: 'sans-serif',
  },
  profileName: {
    fontSize: 16, color: '#E5E7EB', fontWeight: '600', fontFamily: 'sans-serif', textAlign: 'center',
  },
  lockIcon: {
    marginTop: 4,
  },
  createCard: {
    width: '45%', alignItems: 'center', margin: '2.5%', marginTop: 0,
  },
  createCircle: {
    width: 100, height: 100, borderRadius: 50, borderWidth: 2, borderColor: '#374151', borderStyle: 'dashed', justifyContent: 'center', alignItems: 'center', marginBottom: 12,
  },
  createText: {
    fontSize: 16, color: '#9CA3AF', fontWeight: '600', fontFamily: 'sans-serif',
  },
  modalOverlay: {
    flex: 1, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'center', alignItems: 'center',
  },
  modalContent: {
    width: '80%', backgroundColor: '#1E293B', borderRadius: 16, padding: 24,
  },
  modalTitle: {
    fontSize: 20, color: '#F9FAFB', fontWeight: 'bold', marginBottom: 8, fontFamily: 'sans-serif', textAlign: 'center',
  },
  modalSubtitle: {
    fontSize: 14, color: '#9CA3AF', marginBottom: 16, fontFamily: 'sans-serif', textAlign: 'center',
  },
  input: {
    backgroundColor: '#0B0F19', color: '#F9FAFB', borderRadius: 8, padding: 14, fontSize: 16, fontFamily: 'sans-serif', marginBottom: 16,
  },
  pinInput: {
    textAlign: 'center', fontSize: 24, letterSpacing: 8,
  },
  toggleRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16, paddingVertical: 8,
  },
  toggleText: {
    color: '#E5E7EB', fontSize: 16, fontFamily: 'sans-serif',
  },
  checkbox: {
    width: 24, height: 24, borderRadius: 6, borderWidth: 2, borderColor: '#8B5CF6', justifyContent: 'center', alignItems: 'center',
  },
  checkboxActive: {
    backgroundColor: '#8B5CF6',
  },
  modalActions: {
    flexDirection: 'row', justifyContent: 'space-between', marginTop: 8,
  },
  cancelBtn: {
    flex: 1, padding: 14, alignItems: 'center', marginRight: 8, borderRadius: 8, backgroundColor: '#374151',
  },
  cancelBtnText: {
    color: '#E5E7EB', fontSize: 16, fontWeight: '600',
  },
  saveBtn: {
    flex: 1, padding: 14, alignItems: 'center', marginLeft: 8, borderRadius: 8, backgroundColor: '#8B5CF6',
  },
  saveBtnText: {
    color: '#F9FAFB', fontSize: 16, fontWeight: '600',
  },
});

export default ProfileSelectScreen;
