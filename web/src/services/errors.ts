export type ErrorCode =
  | 'unauthorized'
  | 'session_expired'
  | 'incomplete_evidence'
  | 'stale_proposal'
  | 'duplicate_operation'
  | 'insufficient_funds'
  | 'invalid_state'
  | 'validation'
  | 'service_failure'
  | 'integration_blocked';

/** One error shape for every service failure, so screens can show useful recovery text. */
export class ServiceError extends Error {
  readonly code: ErrorCode;
  readonly retryable: boolean;

  constructor(code: ErrorCode, message: string, retryable = false) {
    super(message);
    this.name = 'ServiceError';
    this.code = code;
    this.retryable = retryable;
  }
}

export function toServiceError(error: unknown): ServiceError {
  if (error instanceof ServiceError) return error;
  return new ServiceError('service_failure', 'Something went wrong. Try again.', true);
}
