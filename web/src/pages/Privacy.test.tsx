import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { describe, expect, it } from 'vitest';
import { SignupForm } from './landing/SignupForm';
import { Privacy } from './Privacy';

describe('개인정보 처리방침', () => {
  it('운영자·문의·국외 이전·보유 기간·시행일을 보여 준다', () => {
    render(
      <MemoryRouter>
        <Privacy />
      </MemoryRouter>,
    );
    expect(screen.getByRole('heading', { level: 1, name: '개인정보 처리방침' })).toBeInTheDocument();
    expect(screen.getAllByText(/제때옵스\(예비창업자 허의학\)/).length).toBeGreaterThan(0);
    expect(screen.getAllByRole('link', { name: 'euihakeg@naver.com' })[0]).toHaveAttribute('href', 'mailto:euihakeg@naver.com');
    expect(screen.getByText('Render Services, Inc.(미국)의 클라우드')).toBeInTheDocument();
    expect(screen.getByText('싱가포르')).toBeInTheDocument();
    expect(screen.getByText('7일 뒤 자동 삭제')).toBeInTheDocument();
    expect(screen.getByText('신청일로부터 1년 또는 동의 철회 시 지체 없이 파기')).toBeInTheDocument();
    expect(screen.getByText(/본인 확인 후 10일 이내에 처리합니다/)).toBeInTheDocument();
    expect(screen.getByText(/업로드한 내용이 외부 AI로 전송되지 않습니다/)).toBeInTheDocument();
    expect(screen.getByText('시행일 2026-10-15')).toBeInTheDocument();
  });

  it('보유 기간·가림 범위·통계·접속 기록을 실제 처리와 맞게 알린다', () => {
    render(
      <MemoryRouter>
        <Privacy />
      </MemoryRouter>,
    );
    // 사전 신청 행은 1년이 지나면 마케팅 동의까지 통째로 지운다
    expect(screen.getByText('동의 철회 시 또는 신청일로부터 1년 중 먼저 오는 때까지')).toBeInTheDocument();
    expect(screen.queryByText('동의 철회 시까지')).toBeNull();
    // 가리는 것은 런북 본문·카탈로그 설명이고, 명부는 진단에 그대로 쓰인다
    expect(screen.getByText(/런북 본문과 카탈로그 설명의 연락처·주민번호 등은 자동으로 가립니다/)).toBeInTheDocument();
    expect(screen.getByText(/연락 수단의 종류\(번호·주소는 저장하지 않음\).*모두 결과와 함께 7일 뒤 삭제됩니다/)).toBeInTheDocument();
    expect(screen.getByText(/브라우저 단위로 세며, 개인을 식별하는 정보/)).toBeInTheDocument();
    expect(screen.getByText('IP 주소와 이메일은 접속 기록에 남기지 않습니다.')).toBeInTheDocument();
  });

  it('사전 신청 동의 문구는 국외 이전을 알리고 처리방침으로 이어진다', () => {
    render(
      <MemoryRouter>
        <SignupForm />
      </MemoryRouter>,
    );
    expect(screen.getByText(/싱가포르 서버에 보관/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '개인정보 처리방침' })).toHaveAttribute('href', '/privacy');
    expect(screen.getByText(/목적: 새 기능·사례 소식 안내 · 보유 기간: 동의 철회 시 또는 신청일로부터 1년 중 먼저 오는 때까지/)).toBeInTheDocument();
  });
});
