import { useState, useEffect, useRef } from 'react';
import { Send, X, User, UserPlus } from 'lucide-react';
import axios from 'axios';

const CollaborationChat = ({ psg, patient, onClose }) => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [conversation, setConversation] = useState(null);
  const [doctors, setDoctors] = useState([]);
  const [selectedDoctorId, setSelectedDoctorId] = useState(null);
  const chatBodyRef = useRef(null);

  useEffect(() => {
    const initChat = async () => {
      const token = localStorage.getItem('token');
      try {
        // 1. Get other doctors
        const drsRes = await axios.get('http://localhost:8000/doctors', {
          headers: { Authorization: `Bearer ${token}` }
        });
        setDoctors(drsRes.data);

        // 2. Try to get existing conversation for this file
        const convRes = await axios.get(`http://localhost:8000/conversations/psg/${psg.id}`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (convRes.data && convRes.data.length > 0) {
          setConversation(convRes.data[0]);
          fetchMessages(convRes.data[0].id);
        }
      } catch (err) {
        console.error('Failed to init chat', err);
      }
    };
    initChat();
  }, [psg.id]);

  const fetchMessages = async (convId) => {
    const token = localStorage.getItem('token');
    try {
      const res = await axios.get(`http://localhost:8000/conversations/${convId}/messages`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setMessages(res.data);
    } catch (err) {
      console.error('Failed to fetch messages', err);
    }
  };

  const startConversation = async () => {
    if (!selectedDoctorId) return;
    const token = localStorage.getItem('token');
    try {
      const res = await axios.post(`http://localhost:8000/conversations`, {
        psg_id: psg.id,
        file_type: 'edf',
        target_doctor_id: parseInt(selectedDoctorId, 10)
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setConversation(res.data);
      fetchMessages(res.data.id);
    } catch (err) {
      console.error('Failed to start conversation', err);
    }
  };

  const sendMessage = async () => {
    if (!newMessage.trim() || !conversation) return;
    const token = localStorage.getItem('token');
    try {
      const res = await axios.post(`http://localhost:8000/conversations/${conversation.id}/messages`, {
        content: newMessage
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setMessages([...messages, res.data]);
      setNewMessage('');
    } catch (err) {
      console.error('Failed to send message', err);
    }
  };

  useEffect(() => {
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
    }
  }, [messages]);

  const currentUser = JSON.parse(localStorage.getItem('user') || '{}');

  const otherDoctor = conversation
    ? (conversation.doctor_one.id === currentUser.id ? conversation.doctor_two : conversation.doctor_one)
    : null;

  const getDoctorDisplayName = (dr) => {
    if (!dr) return '';
    return dr.first_name || dr.last_name 
      ? `${dr.first_name || ''} ${dr.last_name || ''}`.trim() 
      : dr.email.split('@')[0];
  };

  return (
    <div className={`chat-modal ${psg ? 'active' : ''}`}>
      <div className="chat-header">
        <div style={{display: 'flex', alignItems: 'center', gap: '10px'}}>
          <div className="logo-dot" style={{width: '6px', height: '6px', animation: 'none', background: '#fff'}}></div>
          <div style={{fontSize: '12px', fontWeight: '700'}}>
             Analyse de {patient.name}
          </div>
        </div>
        <X size={16} onClick={onClose} style={{cursor: 'pointer'}} />
      </div>

      {!conversation ? (
        <div className="chat-body" style={{justifyContent: 'center', textAlign: 'center', padding: '30px'}}>
           <UserPlus size={40} color="var(--text3)" style={{margin: '0 auto 15px'}} />
           <p style={{fontSize: '13px', color: 'var(--text2)', marginBottom: '15px'}}>Demander un avis confraternel sur ce fichier.</p>
           <select 
             className="login-input" 
             style={{fontSize: '12px'}}
             onChange={(e) => setSelectedDoctorId(e.target.value)}
           >
             <option value="">Sélectionner un médecin</option>
             {doctors.filter(d => d.id !== currentUser.id).map(dr => (
               <option key={dr.id} value={dr.id}>Dr. {getDoctorDisplayName(dr)}</option>
             ))}
           </select>
           <button 
             className="btn-login" 
             style={{background: 'var(--red)', marginTop: '10px'}}
             onClick={startConversation}
             disabled={!selectedDoctorId}
           >
             Initier la discussion
           </button>
        </div>
      ) : (
        <>
          <div className="chat-body" ref={chatBodyRef}>
             <div style={{fontSize: '9px', textTransform: 'uppercase', color: 'var(--text3)', textAlign: 'center', marginBottom: '10px', letterSpacing: '1px'}}>
               Discussion avec Dr. {getDoctorDisplayName(otherDoctor) || 'Confrère'}
             </div>
             {messages.map(msg => (
               <div key={msg.id} className={`chat-msg ${msg.sender_id === currentUser.id ? 'sent' : 'received'}`}>
                 <div className="chat-bubble">
                   {msg.content}
                 </div>
                 <div style={{fontSize: '8px', color: 'var(--text3)', marginTop: '4px'}}>
                    {new Date(msg.timestamp || msg.created_at).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})}
                 </div>
               </div>
             ))}
          </div>
          <div className="chat-footer">
            <input 
              type="text" 
              placeholder="Votre message..." 
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            />
            <button onClick={sendMessage}><Send size={16} /></button>
          </div>
        </>
      )}
    </div>
  );
};

export default CollaborationChat;
