import {
  Bell,
  Box,
  CircleUserRound,
  Expand,
  Ghost,
  Hash,
  Hexagon,
  Landmark,
  Mail,
  MapPin,
  MessageCircle,
  MessageSquareText,
  MessagesSquare,
  Phone,
  Power,
  Radio,
  RotateCcw,
  Server,
  Shuffle,
  Siren,
  ToggleRight,
  Users,
  UsersRound,
  Zap,
  type LucideIcon,
} from 'lucide-react';
import type { ActionType, ChannelMedium, NodeKind } from '../../api/types';

export const MEDIUM_ICON: Record<ChannelMedium, LucideIcon> = {
  slack: Hash,
  kakaotalk: MessageCircle,
  teams: UsersRound,
  phone: Phone,
  sms: MessageSquareText,
  email: Mail,
  pager: Siren,
  internal_messenger: MessagesSquare,
  other: Radio,
};

export const ACTION_ICON: Record<ActionType, LucideIcon> = {
  rollback: RotateCcw,
  failover: Shuffle,
  restart: Power,
  feature_flag: ToggleRight,
  scale: Expand,
  other: Zap,
};

export const KIND_ICON: Record<NodeKind | 'ghost', LucideIcon> = {
  service: Box,
  system: Server,
  location: MapPin,
  person: CircleUserRound,
  role: Hexagon,
  team: Users,
  channel: Radio,
  trigger: Bell,
  action: Zap,
  obligation: Landmark,
  ghost: Ghost,
};
