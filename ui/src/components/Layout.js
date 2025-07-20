import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Box,
  IconButton,
  Badge,
  Chip,
  useTheme,
  useMediaQuery,
  Divider,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  CloudUpload as UploadIcon,
  PlayArrow as ProcessingIcon,
  Chat as ChatIcon,
  Analytics as AnalyticsIcon,
  FormatQuote as QuoteIcon,
  Settings as SettingsIcon,
  Notifications as NotificationsIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import { useProcessingStatus, useSystemStatus, useWebSocket } from '../hooks/useWebSocket';

const drawerWidth = 240;

const menuItems = [
  { 
    path: '/dashboard', 
    label: 'Dashboard', 
    icon: <DashboardIcon />,
    description: 'System overview and status'
  },
  { 
    path: '/upload', 
    label: 'Upload & Process', 
    icon: <UploadIcon />,
    description: 'Upload audio files and start processing'
  },
  { 
    path: '/processing', 
    label: 'Processing', 
    icon: <ProcessingIcon />,
    description: 'Monitor processing status and sessions'
  },
  { 
    path: '/chat', 
    label: 'Chat with Podcasts', 
    icon: <ChatIcon />,
    description: 'Interactive chat interface'
  },
  { 
    path: '/analytics', 
    label: 'Analytics', 
    icon: <AnalyticsIcon />,
    description: 'Insights and analytics dashboard'
  },
  { 
    path: '/quotes', 
    label: 'Quote Extractor', 
    icon: <QuoteIcon />,
    description: 'Extract and manage quotes'
  },
  { 
    path: '/settings', 
    label: 'Settings', 
    icon: <SettingsIcon />,
    description: 'System configuration'
  },
];

const Layout = ({ children }) => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  const { connected } = useWebSocket();
  const processingStatus = useProcessingStatus();
  const systemStatus = useSystemStatus();

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const getStatusChip = () => {
    if (!connected) {
      return <Chip label="Offline" color="error" size="small" />;
    }
    if (processingStatus.isProcessing) {
      return <Chip label="Processing" color="warning" size="small" />;
    }
    return <Chip label="Online" color="success" size="small" />;
  };

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
          🎙️ SmartAudioAnalyzer
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Advanced Podcast Analyzer
        </Typography>
      </Box>
      
      <List sx={{ flex: 1, px: 1 }}>
        {menuItems.map((item) => {
          const isActive = location.pathname === item.path;
          const showBadge = item.path === '/processing' && processingStatus.isProcessing;
          
          return (
            <ListItem key={item.path} disablePadding sx={{ mb: 0.5 }}>
              <ListItemButton
                onClick={() => {
                  navigate(item.path);
                  if (isMobile) setMobileOpen(false);
                }}
                selected={isActive}
                sx={{
                  borderRadius: 1,
                  '&.Mui-selected': {
                    backgroundColor: 'primary.main',
                    color: 'primary.contrastText',
                    '&:hover': {
                      backgroundColor: 'primary.dark',
                    },
                    '& .MuiListItemIcon-root': {
                      color: 'primary.contrastText',
                    },
                  },
                }}
              >
                <ListItemIcon>
                  {showBadge ? (
                    <Badge badgeContent="•" color="warning">
                      {item.icon}
                    </Badge>
                  ) : (
                    item.icon
                  )}
                </ListItemIcon>
                <ListItemText 
                  primary={item.label}
                  secondary={!isActive ? item.description : undefined}
                  secondaryTypographyProps={{ 
                    variant: 'caption',
                    sx: { display: { xs: 'none', md: 'block' } }
                  }}
                />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>
      
      <Divider />
      
      <Box sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="caption" color="text.secondary">
            Status
          </Typography>
          {getStatusChip()}
        </Box>
        
        {processingStatus.isProcessing && (
          <Box>
            <Typography variant="caption" color="text.secondary">
              {processingStatus.currentStep && `Step: ${processingStatus.currentStep}`}
            </Typography>
            {processingStatus.progress > 0 && (
              <Typography variant="caption" color="text.secondary" display="block">
                Progress: {processingStatus.progress}%
              </Typography>
            )}
          </Box>
        )}
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
          backgroundColor: 'background.paper',
          color: 'text.primary',
          boxShadow: 1,
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            {menuItems.find(item => item.path === location.pathname)?.label || 'SmartAudioAnalyzer'}
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {processingStatus.isProcessing && (
              <Chip 
                label={`Processing: ${processingStatus.progress}%`} 
                color="warning" 
                size="small" 
                variant="outlined"
              />
            )}
            
            <IconButton color="inherit">
              <Badge badgeContent={0} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      <Box
        component="nav"
        sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
      >
        <Drawer
          variant={isMobile ? 'temporary' : 'permanent'}
          open={isMobile ? mobileOpen : true}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better mobile performance
          }}
          sx={{
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              borderRight: 1,
              borderColor: 'divider',
            },
          }}
        >
          {isMobile && (
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', p: 1 }}>
              <IconButton onClick={handleDrawerToggle}>
                <CloseIcon />
              </IconButton>
            </Box>
          )}
          {drawer}
        </Drawer>
      </Box>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { md: `calc(100% - ${drawerWidth}px)` },
          mt: 8, // AppBar height
          minHeight: 'calc(100vh - 64px)',
          backgroundColor: 'background.default',
        }}
      >
        {children}
      </Box>
    </Box>
  );
};

export default Layout;
