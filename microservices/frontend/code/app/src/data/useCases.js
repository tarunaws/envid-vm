// Envid Metadata (Multimodal only).
const useCases = [
  {
    id: 'envid-metadata-multimodal',
    title: 'Multimodal - Envid metadata',
    cardDescription: 'A separate Envid Metadata backend instance for multimodal experimentation.',
    detailDescription:
      'This use case runs the Envid Metadata UI against a dedicated backend instance so you can change AI model settings without impacting other local demos.',
    image: '/usecases/semantic-search.svg',
    path: '/envid-metadata',
    status: 'available',
    highlights: [
      'Separate backend instance (default port 5016).',
      'Model overrides via microservices/backend/.env.multimodal.local.',
      'Separate local storage index/history from other demos.',
    ],
  },
];

export default useCases;
