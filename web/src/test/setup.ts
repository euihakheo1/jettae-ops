import '@testing-library/jest-dom/vitest';
import * as matchers from '@testing-library/jest-dom/matchers';
import { cleanup } from '@testing-library/react';
import { afterEach, expect } from 'vitest';

// jest-dom 매처를 이 vitest 인스턴스에 확실히 등록한다 (subst 드라이브 경로에서도 동작하도록)
expect.extend(matchers);
afterEach(() => cleanup());
