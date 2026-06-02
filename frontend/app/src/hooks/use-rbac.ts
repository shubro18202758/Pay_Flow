import { hasPermission, rolePolicy, type Permission } from '@/lib/rbac'
import { useUIStore } from '@/stores/use-ui-store'

export function useRoleAccess() {
  const currentRole = useUIStore((s) => s.currentRole)
  const policy = rolePolicy(currentRole)
  return {
    currentRole,
    policy,
    can: (permission: Permission) => hasPermission(currentRole, permission),
  }
}
