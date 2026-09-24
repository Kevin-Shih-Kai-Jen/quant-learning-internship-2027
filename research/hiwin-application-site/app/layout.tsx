import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: '大銀與各種設備的直接連結',
  description: '理解大銀微系統如何讓設備中的工件或工具移動、旋轉並精準停位。',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-Hant">
      <body>{children}</body>
    </html>
  );
}
