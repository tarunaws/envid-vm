import { render, screen } from '@testing-library/react';
import App from './App';

test('renders Envid AI Studio - Dev hero content', () => {
  render(<App />);
  expect(
    screen.getByRole('heading', {
      name: /Build next-gen content ecosystems with Envid AI Studio - Dev/i,
    }),
  ).toBeInTheDocument();

  expect(
    screen.getByText(/AI for media workflows/i),
  ).toBeInTheDocument();
});