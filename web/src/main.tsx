import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource/source-sans-3/400.css';
import '@fontsource/source-sans-3/600.css';
import '@fontsource/source-sans-3/700.css';
import './design/tokens.css';
import './design/base.css';
import './design/components.css';
import App from './App';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
