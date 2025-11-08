#pragma once

/**
 * Social Network Analysis Implementation
 * 
 * Social network analytics and community detection
 */

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <chrono>
#include <memory>

namespace QuasiGraph {

struct Graph; // Forward declaration
struct GraphComponent; // Forward declaration

struct UserProfile {
    size_t user_id;
    std::string username;
    size_t follower_count;
    size_t following_count;
    double influence_score;
    std::vector<size_t> followers;
    std::vector<size_t> following;
    std::vector<std::string> interests;
    std::chrono::system_clock::time_point join_date;
    double activity_level;
};

struct CommunityInfo {
    size_t community_id;
    std::vector<size_t> members;
    double cohesion_score;
    std::vector<std::string> dominant_interests;
    size_t internal_connections;
    size_t external_connections;
    double influence_weight;
    std::string community_type;
};

struct InfluenceMetrics {
    size_t user_id;
    double betweenness_centrality;
    double closeness_centrality;
    double eigenvector_centrality;
    double page_rank_score;
    double katz_centrality;
    size_t reach_estimate;        // Estimated number of reachable users
    double clustering_coefficient;
    std::vector<size_t> influence_sphere;  // Direct influence sphere
};

struct ViralityPrediction {
    size_t content_id;
    size_t content_creator;
    double viral_potential_score;
    size_t estimated_reach;
    std::chrono::milliseconds time_to_peak;
    std::vector<size_t> key_influencers;
    double engagement_rate;
    std::string content_category;
};

struct NetworkMetrics {
    size_t total_users;
    size_t total_connections;
    double average_degree;
    double network_density;
    double clustering_coefficient;
    size_t largest_component_size;
    double small_world_coefficient;
    std::vector<size_t> degree_distribution;
    std::chrono::milliseconds computation_time;
};

enum class AnalysisType {
    COMMUNITY_DETECTION,      // Find communities and clusters
    INFLUENCE_ANALYSIS,       // Identify influential users
    VIRALITY_PREDICTION,      // Predict content virality
    NETWORK_METRICS,          // Calculate network statistics
    ANOMALY_DETECTION,        // Find unusual patterns
    TREND_ANALYSIS           // Analyze trending patterns
};

class SocialNetworkAnalysis {
public:
    /**
     * Constructor
     */
    SocialNetworkAnalysis();
    
    /**
     * Destructor
     */
    ~SocialNetworkAnalysis();
    
    /**
     * Load social network data
     * @param users Vector of user profiles
     * @param connections User connections (follower-following relationships)
     */
    void loadNetworkData(const std::vector<UserProfile>& users,
                        const std::vector<std::pair<size_t, size_t>>& connections);
    
    /**
     * Detect communities in the social network
     * @param min_community_size Minimum size for communities
     * @return Vector of detected communities
     */
    std::vector<CommunityInfo> detectCommunities(size_t min_community_size = 5);
    
    /**
     * Analyze user influence metrics
     * @param user_id Specific user to analyze (0 for all users)
     * @return Influence metrics for specified user(s)
     */
    std::vector<InfluenceMetrics> analyzeInfluence(size_t user_id = 0);
    
    /**
     * Predict content virality
     * @param content_creator User who created the content
     * @param content_type Type of content
     * @param initial_engagement Initial engagement metrics
     * @return Virality prediction
     */
    ViralityPrediction predictVirality(size_t content_creator,
                                      const std::string& content_type,
                                      double initial_engagement = 0.0);
    
    /**
     * Calculate comprehensive network metrics
     * @return Network statistics and metrics
     */
    NetworkMetrics calculateNetworkMetrics();
    
    /**
     * Find influential users for marketing/targeting
     * @param target_count Number of influencers to find
     * @param criteria Selection criteria (reach, engagement, etc.)
     * @return Top influential users
     */
    std::vector<size_t> findTopInfluencers(size_t target_count = 10,
                                          const std::string& criteria = "combined");
    
    /**
     * Recommend connections for users
     * @param user_id User to recommend connections for
     * @param recommendation_count Number of recommendations
     * @return Recommended users to connect with
     */
    std::vector<size_t> recommendConnections(size_t user_id, size_t recommendation_count = 5);
    
    /**
     * Detect anomalous behavior patterns
     * @return Users with suspicious activity patterns
     */
    std::vector<size_t> detectAnomalies();
    
    /**
     * Analyze trending topics and patterns
     * @param time_window Time window for trend analysis
     * @return Trending topics and their metrics
     */
    std::vector<std::pair<std::string, double>> analyzeTrends(
        std::chrono::hours time_window = std::chrono::hours(24));
    
    /**
     * Simulate information spread through network
     * @param source_user User who starts the spread
     * @param spread_probability Probability of transmission per edge
     * @param max_steps Maximum simulation steps
     * @return Users reached and spread pattern
     */
    std::vector<size_t> simulateInformationSpread(size_t source_user,
                                                  double spread_probability = 0.1,
                                                  size_t max_steps = 10);
    
    /**
     * Set analysis parameters
     * @param time_limit Maximum time for analysis
     * @param precision_level Analysis precision (0.0 to 1.0)
     * @param use_quasi_algorithms Whether to use quasi-polynomial algorithms
     */
    void setParameters(std::chrono::milliseconds time_limit = std::chrono::milliseconds(60000),
                      double precision_level = 0.95,
                      bool use_quasi_algorithms = true);
    
    /**
     * Get analysis performance statistics
     */
    struct AnalysisStats {
        size_t total_analyses;
        std::chrono::milliseconds total_time;
        double average_precision;
        size_t average_community_count;
        double success_rate;
    };
    
    AnalysisStats getStats() const;
    
    /**
     * Reset statistics
     */
    void resetStats();

private:
    std::vector<UserProfile> users_;
    std::vector<std::pair<size_t, size_t>> connections_;
    std::unique_ptr<Graph> network_graph_;
    
    std::chrono::milliseconds time_limit_;
    double precision_level_;
    bool use_quasi_algorithms_;
    
    mutable AnalysisStats stats_;
    
    // Core analysis algorithms
    std::vector<CommunityInfo> detectCommunitiesQuasiPolynomial();
    std::vector<CommunityInfo> detectCommunitiesModularity();
    std::vector<CommunityInfo> detectCommunitiesSpectral();
    
    std::vector<InfluenceMetrics> calculateInfluenceMetricsQuasiPolynomial();
    std::vector<InfluenceMetrics> calculateInfluenceMetricsTraditional();
    
    // Centrality calculations
    double calculateBetweennessCentrality(size_t user_id);
    double calculateClosenessCentrality(size_t user_id);
    double calculateEigenvectorCentrality(size_t user_id);
    double calculatePageRank(size_t user_id);
    double calculateKatzCentrality(size_t user_id);
    
    // Community detection utilities
    std::vector<std::vector<size_t>> findConnectedComponents();
    double calculateModularity(const std::vector<std::vector<size_t>>& communities);
    std::vector<std::vector<size_t>> optimizeCommunities(
        const std::vector<std::vector<size_t>>& initial_communities);
    double calculateCommunityInfluence(const CommunityInfo& community);
    std::vector<std::string> getDominantInterests(const CommunityInfo& community);
    
    // Virality prediction components
    double calculateViralPotential(size_t content_creator, const std::string& content_type);
    std::vector<size_t> identifyKeyInfluencers(size_t content_creator, size_t count = 5);
    size_t estimateReach(size_t source_user, double spread_probability);
    std::chrono::milliseconds predictTimeToPeak(size_t content_creator, const std::string& content_type);
    double estimateEngagementRate(size_t content_creator, const std::string& content_type, double base_engagement);
    double getContentViralityModifier(const std::string& content_type);
    double calculateNetworkViralityModifier();
    double calculateActivityViralityModifier(size_t user_id);
    std::vector<size_t> calculateInfluenceSphere(size_t user_id);
    double predictInitialEngagement(size_t content_creator, const std::string& content_type);
    double calculateLocalClusteringCoefficient(size_t user_id);
    std::vector<size_t> calculateShortestPaths(size_t source_user);
    size_t generateContentId();
    bool isOnShortestPath(size_t source, size_t target, size_t intermediate);
    double calculateCommunityCohesion(const CommunityInfo& community);
    size_t countInternalConnections(const CommunityInfo& community);
    size_t countExternalConnections(const CommunityInfo& community);
    double calculateComponentCohesion(const GraphComponent& component);
    std::vector<CommunityInfo> optimizeCommunityStructure(const std::vector<CommunityInfo>& communities);
    bool shouldMergeCommunities(const CommunityInfo& c1, const CommunityInfo& c2);
    
    // Recommendation system
    std::vector<size_t> recommendBySimilarity(size_t user_id, size_t count);
    std::vector<size_t> recommendByInfluence(size_t user_id, size_t count);
    std::vector<size_t> recommendByCommunity(size_t user_id, size_t count);
    double calculateUserSimilarity(size_t user1, size_t user2);
    
    // Anomaly detection
    bool isAnomalousUser(size_t user_id);
    double calculateAnomalyScore(size_t user_id);
    std::vector<double> getActivityPattern(size_t user_id);
    
    // Trend analysis
    std::unordered_map<std::string, double> calculateTopicTrends(
        std::chrono::hours time_window);
    std::vector<std::string> getTrendingTopics(std::chrono::hours time_window);
    
    // Information spread simulation
    std::vector<size_t> simulateSpreadQuasiPolynomial(size_t source_user, 
                                                      double probability, size_t steps);
    std::vector<size_t> simulateSpreadIndependentCascade(size_t source_user, 
                                                         double probability, size_t steps);
    std::vector<size_t> simulateSpreadLinearThreshold(size_t source_user, 
                                                      double probability, size_t steps);
    
    // Network metrics calculation
    double calculateAveragePathLength();
    double calculateClusteringCoefficient();
    double calculateSmallWorldCoefficient();
    std::vector<size_t> getDegreeDistribution();
    
    // Graph construction and management
    void buildNetworkGraph();
    void updateNetworkGraph();
    std::vector<size_t> getNeighbors(size_t user_id);
    size_t getDegree(size_t user_id);
    
    // Performance optimization
    std::unordered_map<size_t, std::vector<size_t>> neighbor_cache_;
    std::unordered_map<size_t, double> centrality_cache_;
    bool cache_valid_;
    
    void updateCache();
    void clearCache();
    
    // Quasi-polynomial algorithm utilities
    std::vector<size_t> findInfluentialSetQuasiPolynomial(size_t target_size);
    std::vector<std::vector<size_t>> decomposeNetworkQuasiPolynomial();
    double estimateInfluenceSpreadQuasiPolynomial(const std::vector<size_t>& seed_set);
};

/**
 * Factory class for creating specialized social network analyzers
 */
class SocialNetworkFactory {
public:
    /**
     * Create analyzer for Twitter-like networks
     */
    static std::unique_ptr<SocialNetworkAnalysis> createTwitterAnalyzer();
    
    /**
     * Create analyzer for Facebook-like networks
     */
    static std::unique_ptr<SocialNetworkAnalysis> createFacebookAnalyzer();
    
    /**
     * Create analyzer for LinkedIn-like professional networks
     */
    static std::unique_ptr<SocialNetworkAnalysis> createLinkedInAnalyzer();
    
    /**
     * Create analyzer for Instagram-like visual networks
     */
    static std::unique_ptr<SocialNetworkAnalysis> createInstagramAnalyzer();
    
    /**
     * Create analyzer for research citation networks
     */
    static std::unique_ptr<SocialNetworkAnalysis> createCitationNetworkAnalyzer();
};

} // namespace QuasiGraph
