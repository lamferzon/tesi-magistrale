classdef FSIM

    properties(Access = public)
        Y_sim
        X {mustBeNumeric}
        params
        n_basis
        points
        n_iter {mustBeInteger, mustBePositive, mustBeNonzero}
    end

    properties(Access = private)
        n {mustBeNumeric, mustBePositive, mustBeNonzero}
        q {mustBeNumeric, mustBePositive, mustBeNonzero}
        b {mustBeNumeric, mustBePositive, mustBeNonzero}
        T {mustBeNumeric, mustBePositive, mustBeNonzero}
        dist_mat {mustBeNumeric}
        sigma_eps {mustBeNumeric}
        sigma_eta {mustBeNumeric}
    end
    
    methods (Access = public)

        function obj = FSIM(points, q, T, X, params, n_basis)

            arguments
                points = FSIM.get_points()
                q (1, 1) {mustBeNumeric, mustBePositive} = 24
                T (1, 1) {mustBeNumeric, mustBePositive} = 365
                X.XData {mustBeNumeric} = ones(size(points, 1)*q, 1)
                params.Parameters = struct()
                n_basis.NumBasis = struct()
            end

            % parameters setting
            obj.n = size(points, 1);
            obj.q = q;
            obj.b = size(X.XData, 2);
            obj.T = T;
            obj.X = X.XData;
            obj.n_basis = obj.check_n_basis(n_basis.NumBasis);
            obj.params = obj.check_params(params.Parameters);
            obj.points = points;

            % computing distance matrix
            obj.dist_mat = obj.get_dist_mat();
            
            % summary
            disp("FSIM (Functional SIMulator)")
            disp(" ")
            disp("Authors: ")
            disp("- Lorenzo LEONI")
            disp("- Nicola ZAMBELLI")
            disp(" ")
            disp("Dimensions: ")
            disp("- n, q, b and T: [" + num2str(obj.n) + " " + ... 
                num2str(obj.q) + " " + num2str(obj.b) + " " + num2str(obj.T) + ...
                "]")
            disp("- p_eps, p_beta and p_z: [" + num2str(obj.n_basis.p_eps) + ...
                " " + num2str(obj.n_basis.p_beta) + " " + ... 
                num2str(obj.n_basis.p_z) + "]")
            disp(" ")
            disp("Parameters set: ")
            disp("- rho: " + num2str(obj.params.rho))
            disp("- c_eps: [" + num2str(obj.params.c_eps') + "]")
            for i=1:obj.b
                disp("- c_beta_" + num2str(i) + ... 
                    ": [" + num2str(obj.params.c_beta(i, :)) + "]")
            end
            disp("- diag_G: [" + num2str(obj.params.diag_G') + "]")
            disp("- diag_V: [" + num2str(obj.params.diag_V') + "]")
            disp("- theta: [" + num2str(obj.params.theta') + "]")

        end

        function obj = run(obj, n_iter)

            arguments
                obj {mustBeNonempty}
                n_iter (1, 1) {mustBeInteger, mustBePositive, mustBeNonzero} = 1
            end
            
            % parameters setting
            obj.Y_sim = cell(n_iter, 1);
            obj.n_iter = n_iter;

            % computing sigma_eps
            %obj.sigma_eps = sparse(obj.get_sigma_eps());
            
            % computing sigma_eta
            obj.sigma_eta = sparse(obj.get_sigma_eta());


            for iter=1:n_iter
                Y_sim = zeros(obj.n*obj.q, obj.T);
                Z = obj.simulate_AR1();
                for t=1:obj.T
                    disp("TO DO")
                end
            end

        end

    end

    methods (Access = private)
        
        function params = check_params(obj, params)

            % setting rho
            if ~isfield(params, "rho")
                params.rho = randi([1 50]);
            end

            % setting c_eps
            if ~isfield(params, "c_eps")
                params.c_eps = randi([1 100], obj.n_basis.p_eps, 1);
            end

            % setting c_beta
            if ~isfield(params, "c_beta")
                params.c_beta = randi([1 100], obj.b, obj.n_basis.p_beta);
            end
            
            % setting z_0
            if ~isfield(params, "z_0")
                mu_0 = zeros(obj.n*obj.n_basis.p_z, 1);
                sigma_0 = speye(obj.n*obj.n_basis.p_z);
                params.z_0 = mvnrnd(mu_0, sigma_0, 1)';
            end

            % setting G
            if ~isfield(params, "diag_G")
                params.diag_G = rand(obj.n_basis.p_z, 1);
            end

            % setting V
            if ~isfield(params, "diag_V")
                params.diag_V = randi([1 10], obj.n_basis.p_z, 1);
            end

            % setting theta
            if ~isfield(params, "theta")
                params.theta = repmat(randi([1 50]), obj.n_basis.p_z, 1);
            end

        end

        function n_basis = check_n_basis(~, n_basis)
            
            % setting p_eps
            if ~isfield(n_basis, "p_eps")
                n_basis.p_eps = randi([2 10]);
            end

            % setting p_beta
            if ~isfield(n_basis, "p_beta")
                n_basis.p_beta = randi([2 10]);
            end

            % setting p_z
            if ~isfield(n_basis, "p_z")
                n_basis.p_z = randi([2 10]);
            end

        end

        function dist_mat = get_dist_mat(obj)

            dist_mat = zeros(obj.n);
            for i=1:obj.n
                dist_mat(i, i+1:end) = distance(obj.points{i, :}, ...
                    obj.points{i+1:end, :});
            end
            dist_mat = dist_mat + dist_mat';

        end

        function sigma_eta = get_sigma_eta(obj)

            sigma_eta = zeros(obj.n*obj.n_basis.p_z);
            for i=1:obj.n_basis.p_z
                block_i = obj.params.diag_V(i).*exp(-obj.dist_mat/obj.params.theta(i));
                sigma_eta((i-1)*obj.n+1:i*obj.n, (i-1)*obj.n+1:i*obj.n) = block_i;
            end

        end

        function Z = simulate_AR1(obj)
            
            Z = zeros(obj.n*obj.n_basis.p_z, obj.T+1);
            Z(:, 1) = obj.params.z_0;

            % computing diag_G_tilde
            diag_G_tilde = repelem(obj.params.diag_G, obj.n);

            % AR1 simulation
            for t=2:obj.T
                Z(:, t) = diag_G_tilde.*Z(:, t-1) + mvnrnd(zeros(obj.n*obj.n_basis.p_z, 1), ...
                    obj.sigma_eta)';
            end

        end

    end

    methods (Static)
        
        function coord = get_points(n, lon_range, lat_range, step)

            arguments
                n (1, 1) {mustBeNumeric, mustBePositive} = 10
                lon_range.LonRange (1, 2) {mustBeNumeric} = [8.5, 10.5]
                lat_range.LatRange (1, 2) {mustBeNumeric} = [44.5, 46.5]
                step.Step (1, 1) {mustBeNumeric} = .007
            end

            % longitude range check
            if lon_range.LonRange(1) < -180 || lon_range.LonRange(2) > 180 ...
                    || lon_range.LonRange(1) > lon_range.LonRange(2)
                error("Not valid longitude range.")
            end

            % latitude range check
            if lat_range.LatRange(1) < -90 || lat_range.LatRange(2) > 90 ...
                    || lat_range.LatRange(1) > lat_range.LatRange(2)
                error("Not valid latitude range.")
            end
            
            % grid generation
            [lon, lat] = meshgrid(lon_range.LonRange(1):step.Step:lon_range.LonRange(2), ...
                lat_range.LatRange(1):step.Step:lat_range.LatRange(2));
            coord = [lon(:) lat(:)];

            % random points extraction
            coord = array2table(coord(randperm(length(coord), n), :));
            coord.Properties.VariableNames = ["Longitude", "Latitude"];

        end

    end

end

