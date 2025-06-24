import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
from datetime import datetime

class MetricsTracker:
    """Track and visualize training metrics."""
    
    def __init__(self, save_dir="training_results"):
        """Initialize metrics tracker.
        
        Args:
            save_dir: Directory to save plots and reports
        """
        self.save_dir = save_dir
        self.episode_rewards = []
        self.training_losses = []
        self.loss_steps = []
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def add_episode_reward(self, reward):
        """Add episode reward to tracking."""
        self.episode_rewards.append(reward)
        
    def add_training_loss(self, loss, step):
        """Add training loss to tracking."""
        self.training_losses.append(loss)
        self.loss_steps.append(step)
        
    def plot_episode_rewards(self, save_path=None):
        """Plot episode rewards over time."""
        if not self.episode_rewards:
            return None
            
        plt.figure(figsize=(12, 6))
        episodes = list(range(1, len(self.episode_rewards) + 1))
        
        plt.plot(episodes, self.episode_rewards, 'b-', alpha=0.7, linewidth=1)
        
        # Add moving average for smoother visualization
        if len(self.episode_rewards) >= 10:
            window_size = min(100, len(self.episode_rewards) // 10)
            moving_avg = []
            for i in range(len(self.episode_rewards)):
                start_idx = max(0, i - window_size + 1)
                avg = np.mean(self.episode_rewards[start_idx:i+1])
                moving_avg.append(avg)
            plt.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window_size} episodes)')
            plt.legend()
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards Over Time')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return plt.gcf()
    
    def plot_training_loss(self, save_path=None):
        """Plot training loss over time."""
        if not self.training_losses:
            return None
            
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.loss_steps, self.training_losses, 'g-', alpha=0.7, linewidth=1)
        
        # Add moving average for smoother visualization
        if len(self.training_losses) >= 10:
            window_size = min(1000, len(self.training_losses) // 10)
            moving_avg = []
            for i in range(len(self.training_losses)):
                start_idx = max(0, i - window_size + 1)
                avg = np.mean(self.training_losses[start_idx:i+1])
                moving_avg.append(avg)
            plt.plot(self.loss_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window_size} steps)')
            plt.legend()
        
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return plt.gcf()
    
    def generate_report(self, filename=None):
        """Generate PDF report with all plots."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_report_{timestamp}.pdf"
        
        filepath = os.path.join(self.save_dir, filename)
        
        with PdfPages(filepath) as pdf:
            # Page 1: Episode Rewards
            fig1 = self.plot_episode_rewards()
            if fig1:
                pdf.savefig(fig1, bbox_inches='tight')
                plt.close(fig1)
            
            # Page 2: Training Loss
            fig2 = self.plot_training_loss()
            if fig2:
                pdf.savefig(fig2, bbox_inches='tight')
                plt.close(fig2)
            
            # Page 3: Summary Statistics
            self._create_summary_page(pdf)
        
        print(f"Training report saved to: {filepath}")
        return filepath
    
    def _create_summary_page(self, pdf):
        """Create summary statistics page."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Summary Statistics', fontsize=16)
        
        # Episode rewards statistics
        if self.episode_rewards:
            rewards_array = np.array(self.episode_rewards)
            
            ax1.hist(rewards_array, bins=30, alpha=0.7, color='blue')
            ax1.set_title('Episode Rewards Distribution')
            ax1.set_xlabel('Reward')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Statistics text
            stats_text = f"""
            Total Episodes: {len(self.episode_rewards)}
            Mean Reward: {rewards_array.mean():.2f}
            Std Reward: {rewards_array.std():.2f}
            Min Reward: {rewards_array.min():.2f}
            Max Reward: {rewards_array.max():.2f}
            """
            ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=12,
                    verticalalignment='center', fontfamily='monospace')
            ax2.set_title('Episode Statistics')
            ax2.axis('off')
        
        # Training loss statistics
        if self.training_losses:
            loss_array = np.array(self.training_losses)
            
            ax3.hist(loss_array, bins=30, alpha=0.7, color='green')
            ax3.set_title('Training Loss Distribution')
            ax3.set_xlabel('Loss')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
            
            # Loss statistics text
            loss_stats_text = f"""
            Total Training Steps: {len(self.training_losses)}
            Mean Loss: {loss_array.mean():.4f}
            Std Loss: {loss_array.std():.4f}
            Min Loss: {loss_array.min():.4f}
            Max Loss: {loss_array.max():.4f}
            """
            ax4.text(0.1, 0.5, loss_stats_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='center', fontfamily='monospace')
            ax4.set_title('Loss Statistics')
            ax4.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def save_plots(self):
        """Save individual plots as PNG files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save episode rewards plot
        rewards_path = os.path.join(self.save_dir, f"episode_rewards_{timestamp}.png")
        self.plot_episode_rewards(rewards_path)
        
        # Save training loss plot
        loss_path = os.path.join(self.save_dir, f"training_loss_{timestamp}.png")
        self.plot_training_loss(loss_path)
        
        return rewards_path, loss_path
