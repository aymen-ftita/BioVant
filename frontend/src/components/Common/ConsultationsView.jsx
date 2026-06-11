import { useState, useEffect, useRef } from 'react';
import { Send, UserPlus, MessageSquare, Clock } from 'lucide-react';
import axios from 'axios';
import './ConsultationsView.css';

const ConsultationsView = () => {
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const chatBodyRef = useRef(null);

  const currentUser = JSON.parse(localStorage.getItem('user') || '{}');

  useEffect(() => {
    fetchConversations();
  }, []);

  const fetchConversations = async () => {
    const token = localStorage.getItem('token');
    try {
      const res = await axios.get('http://localhost:8000/conversations', {
        headers: { Authorization: `Bearer ${token}` }
      });
      // Sort by recently created (or you could sort by latest message if backend returned it)
      const sorted = res.data.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      setConversations(sorted);
    } catch (err) {
      console.error('Failed to fetch conversations', err);
    }
  };

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

  const handleSelectConversation = (convId) => {
    setActiveConversationId(convId);
    fetchMessages(convId);
  };

  const sendMessage = async () => {
    if (!newMessage.trim() || !activeConversationId) return;
    const token = localStorage.getItem('token');
    try {
      const res = await axios.post(`http://localhost:8000/conversations/${activeConversationId}/messages`, {
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

  const getDoctorDisplayName = (dr) => {
    if (!dr) return 'Confrère';
    return dr.first_name || dr.last_name 
      ? `${dr.first_name || ''} ${dr.last_name || ''}`.trim() 
      : dr.email.split('@')[0];
  };

  const activeConv = conversations.find(c => c.id === activeConversationId);
  const activeOtherDoctor = activeConv
    ? (activeConv.doctor_one.id === currentUser.id ? activeConv.doctor_two : activeConv.doctor_one)
    : null;

  return (
    <div className="consultations-layout">
      {/* LEFT: CONVERSATIONS LIST */}
      <div className="consultations-sidebar">
        <div className="consultations-header">
          <MessageSquare size={18} />
          <h3>Mes Consultations</h3>
        </div>
        
        <div className="consultations-list">
          {conversations.length === 0 ? (
            <div className="empty-state-list">
              Aucune conversation pour le moment.
            </div>
          ) : (
            conversations.map(conv => {
              const isSelected = conv.id === activeConversationId;
              const otherDr = conv.doctor_one.id === currentUser.id ? conv.doctor_two : conv.doctor_one;
              const dateObj = new Date(conv.created_at);
              const isToday = dateObj.toDateString() === new Date().toDateString();
              const timeString = isToday 
                ? dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                : dateObj.toLocaleDateString([], { day: '2-digit', month: 'short' });

              return (
                <div 
                  key={conv.id} 
                  className={`consultation-item ${isSelected ? 'active' : ''}`}
                  onClick={() => handleSelectConversation(conv.id)}
                >
                  <div className="avatar-circle">
                    {getDoctorDisplayName(otherDr)[0].toUpperCase()}
                  </div>
                  <div className="consultation-info">
                    <div className="consultation-top">
                      <span className="dr-name">Dr. {getDoctorDisplayName(otherDr)}</span>
                      <span className="conv-time">{timeString}</span>
                    </div>
                    <div className="consultation-bottom">
                      <span className="conv-subject">Dossier PSG #{conv.psg_id}</span>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* RIGHT: CHAT VIEW */}
      <div className="consultations-main">
        {!activeConversationId ? (
          <div className="empty-chat-state">
            <UserPlus size={48} color="var(--border)" />
            <h2>Sélectionnez une conversation</h2>
            <p>Choisissez un avis confraternel dans la liste de gauche pour lire ou envoyer des messages.</p>
          </div>
        ) : (
          <div className="chat-window">
            <div className="chat-window-header">
              <div className="avatar-circle small">
                {getDoctorDisplayName(activeOtherDoctor)[0].toUpperCase()}
              </div>
              <div className="chat-header-info">
                <div className="dr-name">Dr. {getDoctorDisplayName(activeOtherDoctor)}</div>
                <div className="subject">Concerne: Examen PSG #{activeConv.psg_id}</div>
              </div>
            </div>

            <div className="chat-body full-height" ref={chatBodyRef}>
              <div style={{fontSize: '9px', textTransform: 'uppercase', color: 'var(--text3)', textAlign: 'center', marginBottom: '15px', letterSpacing: '1px'}}>
                Historique de la discussion
              </div>
              
              {messages.length === 0 ? (
                <div style={{textAlign: 'center', color: 'var(--text3)', fontSize: '11px', marginTop: '20px'}}>
                  Aucun message. Soyez le premier à écrire !
                </div>
              ) : (
                messages.map(msg => {
                  const isSentByMe = msg.sender_id === currentUser.id;
                  return (
                    <div key={msg.id} className={`chat-msg ${isSentByMe ? 'sent' : 'received'}`}>
                      <div className="chat-bubble">
                        {msg.content}
                      </div>
                      <div style={{fontSize: '8px', color: 'var(--text3)', marginTop: '4px'}}>
                        {new Date(msg.timestamp || msg.created_at).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})}
                      </div>
                    </div>
                  );
                })
              )}
            </div>

            <div className="chat-window-footer">
              <input 
                type="text" 
                placeholder="Écrivez votre message ici..." 
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
              />
              <button className="btn-send" onClick={sendMessage} disabled={!newMessage.trim()}>
                <Send size={16} />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ConsultationsView;
