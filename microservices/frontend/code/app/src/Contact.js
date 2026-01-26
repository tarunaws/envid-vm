import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

const Page = styled.section`
  max-width: 960px;
  margin: 0 auto;
  padding: 2rem 1.25rem;
  color: #e5e5e5;
`;

const Title = styled.h1`
  color: #ffffff;
  font-weight: 800;
  margin: 0 0 0.75rem 0;
  font-size: clamp(1.5rem, 3vw, 2rem);
  text-align: center;
`;

const Lead = styled.p`
  text-align: center;
  margin: 0 auto 1.25rem;
  max-width: 760px;
  line-height: 1.7;
`;

const Form = styled.form`
  background: #1f1f1f;
  border: 1px solid #333;
  border-radius: 10px;
  padding: 1.25rem;
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
  @media (max-width: 640px) {
    grid-template-columns: 1fr;
  }
`;

const Field = styled.div`
  display: flex;
  flex-direction: column;
`;

const Label = styled.label`
  font-size: 0.95rem;
  margin-bottom: 0.35rem;
  color: #ffffff;
`;

const Input = styled.input`
  background: #141414;
  border: 1px solid #333;
  border-radius: 6px;
  padding: 0.8rem 0.9rem;
  color: #e5e5e5;
  font-size: 1rem;
`;

const Textarea = styled.textarea`
  background: #141414;
  border: 1px solid #333;
  border-radius: 6px;
  padding: 0.8rem 0.9rem;
  color: #e5e5e5;
  font-size: 1rem;
`;

const Select = styled.select`
  background: #141414;
  border: 1px solid #333;
  border-radius: 6px;
  padding: 0.8rem 0.9rem;
  color: #e5e5e5;
  font-size: 1rem;
  cursor: pointer;
  option {
    background: #141414;
    color: #e5e5e5;
  }
`;

const Submit = styled.button`
  margin-top: 1rem;
  padding: 0.9rem 1.2rem;
  background: #e50914;
  color: #ffffff;
  font-weight: 800;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s ease;
  &:hover { background: #f6121d; }
`;

// Removed inline Notice in favor of Toast

const Toast = styled.div`
  position: fixed;
  right: 20px;
  bottom: 20px;
  min-width: 260px;
  max-width: 420px;
  background: ${({ $ok }) => ($ok ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)')};
  color: ${({ $ok }) => ($ok ? '#86efac' : '#fca5a5')};
  border: 1px solid ${({ $ok }) => ($ok ? '#16a34a' : '#dc2626')};
  border-left: 4px solid ${({ $ok }) => ($ok ? '#22c55e' : '#ef4444')};
  padding: 0.9rem 1rem;
  border-radius: 8px;
  box-shadow: 0 10px 24px rgba(0,0,0,0.35);
  z-index: 1000;
`;

const ToastHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.35rem;
  font-weight: 800;
  color: #fff;
`;

const ToastClose = styled.button`
  background: transparent;
  border: none;
  color: inherit;
  cursor: pointer;
  font-size: 1.15rem;
  line-height: 1;
`;

// List of countries for the dropdown
const countries = [
  'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Argentina', 'Armenia', 'Australia',
  'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium',
  'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei',
  'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde',
  'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo',
  'Costa Rica', 'Croatia', 'Cuba', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica',
  'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia',
  'Eswatini', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany',
  'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras',
  'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica',
  'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia',
  'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Madagascar',
  'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius',
  'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique',
  'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria',
  'North Korea', 'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Panama', 'Papua New Guinea',
  'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda',
  'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino',
  'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore',
  'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea', 'South Sudan',
  'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan',
  'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
  'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States',
  'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican City', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe'
];


export default function Contact() {
  const API_BASE = process.env.REACT_APP_API_BASE || '';
  const [form, setForm] = useState({
    firstName: '', lastName: '', company: '', email: '', phone: '', country: '', comments: ''
  });
  const [status, setStatus] = useState({ loading: false, ok: null, msg: '' });
  const [showToast, setShowToast] = useState(false);

  useEffect(() => {
    if (status.msg) {
      setShowToast(true);
      const t = setTimeout(() => setShowToast(false), 4000);
      return () => clearTimeout(t);
    }
  }, [status.msg]);

  const onChange = (e) => setForm({ ...form, [e.target.name]: e.target.value });

  const onSubmit = async (e) => {
    e.preventDefault();
    // Minimal client-side validation
    if (!form.firstName || !form.lastName || !form.email) {
      setStatus({ loading: false, ok: false, msg: 'Please fill in First Name, Last Name, and Email.' });
      return;
    }
    try {
      setStatus({ loading: true, ok: null, msg: '' });
      const url = API_BASE ? `${API_BASE}/contact` : '/contact';
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form)
      });
      const json = await res.json();
      if (res.ok && json.ok) {
        setStatus({ loading: false, ok: true, msg: 'Thanks! Your message has been sent.' });
        setForm({ firstName: '', lastName: '', company: '', email: '', phone: '', country: '', comments: '' });
      } else {
        throw new Error(json.error || 'Failed to submit');
      }
    } catch (err) {
      setStatus({ loading: false, ok: false, msg: err.message || 'Something went wrong.' });
    }
  };

  return (
    <Page>
      <Title>Contact Us</Title>
      <Lead>
        We’d love to hear from you! Whether you’re interested in our demos, training, careers, or partnerships,
        please reach out using the form below.
      </Lead>

      <Form onSubmit={onSubmit} className="contact-form">
        <Grid>
          <Field>
            <Label htmlFor="firstName">First Name</Label>
            <Input id="firstName" name="firstName" type="text" required value={form.firstName} onChange={onChange} />
          </Field>
          <Field>
            <Label htmlFor="lastName">Last Name</Label>
            <Input id="lastName" name="lastName" type="text" required value={form.lastName} onChange={onChange} />
          </Field>
          <Field>
            <Label htmlFor="company">Company</Label>
            <Input id="company" name="company" type="text" value={form.company} onChange={onChange} />
          </Field>
          <Field>
            <Label htmlFor="email">Email</Label>
            <Input id="email" name="email" type="email" required value={form.email} onChange={onChange} />
          </Field>
          <Field>
            <Label htmlFor="phone">Phone</Label>
            <Input id="phone" name="phone" type="tel" value={form.phone} onChange={onChange} />
          </Field>
          <Field>
            <Label htmlFor="country">Country</Label>
            <Select id="country" name="country" value={form.country} onChange={onChange}>
              <option value="">Select a country</option>
              {countries.map((country) => (
                <option key={country} value={country}>
                  {country}
                </option>
              ))}
            </Select>
          </Field>
          <Field style={{ gridColumn: '1 / -1' }}>
            <Label htmlFor="comments">Comments / Inquiry Description</Label>
            <Textarea id="comments" name="comments" rows="4" value={form.comments} onChange={onChange} />
          </Field>
        </Grid>
        <Submit type="submit" disabled={status.loading}>{status.loading ? 'Submitting…' : 'Submit'}</Submit>
      </Form>
      {/* Inline notice replaced by toast for better visibility */}
      {showToast && status.msg && (
        <Toast role="status" aria-live="polite" $ok={!!status.ok} onClick={() => setShowToast(false)}>
          <ToastHeader>
            <span>{status.ok ? 'Success' : 'Notice'}</span>
            <ToastClose aria-label="Close">×</ToastClose>
          </ToastHeader>
          <div>{status.msg}</div>
        </Toast>
      )}

      {/* Additional inquiries section removed per request */}
    </Page>
  );
}
