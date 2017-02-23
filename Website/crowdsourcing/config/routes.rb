Rails.application.routes.draw do
  # The priority is based upon order of creation: first created -> highest priority.
  # See how all your routes lay out with "rake routes".

  # You can have the root of your site routed with "root"
  # root 'welcome#index'
  root to: 'pages#index', campaign: nil, userid: nil, mw: nil
  get 'pages/index/:user/:mw' => 'pages#index', user: :user, mw: :mw
  # Example of regular route:
  #   get 'products/:id' => 'catalog#view'
  post 'crowdsourceuser' => 'users#crowdsourceuser'
  get 'newuser' => 'users#new'
  post 'newuser' => 'users#create'
  get 'about' => 'pages#about'

  get 'intro_part1' => 'pages#intro_part1'
  get 'totraining_part1' => 'pages#totraining_part1'
  get 'training_part1' => 'pages#training_part1'
  get 'golden' => 'pages#golden'
  get 'submit_golden' => 'pages#submit_golden'

  get 'ready_part1' => 'pages#ready_part1'
  get 'show_part1' => 'scores#show_part1'
  get 'checkimage_part1' => 'scores#checkimage_part1'
  get 'contentcheck_part1' => 'scores#contentcheck_part1'

  get 'intro_part2' => 'pages#intro_part2'
  get 'totraining_part2' => 'pages#totraining_part2'
  get 'training_part2' => 'pages#training_part2'

  get 'ready_part2' => 'pages#ready_part2'
  get 'show_part2' => 'scores#show_part2'
  get 'checkimage_part2' => 'scores#checkimage_part2'
  get 'contentcheck_part2' => 'scores#contentcheck_part2'

  post 'update' => 'users#update'
  get 'content' => 'users#content'
  get 'contentanswer' => 'users#contentanswer'

  get 'end' => 'pages#end'
  get 'endwrong' => 'pages#endwrong'
  get 'logout' => 'pages#logout'

  # Example of named route that can be invoked with purchase_url(id: product.id)
  #   get 'products/:id/purchase' => 'catalog#purchase', as: :purchase

  # Example resource route (maps HTTP verbs to controller actions automatically):
  #   resources :products

  resources :scores

  resources :users

  resources :campaignsets

  resources :images
  # Example resource route with options:
  #   resources :products do
  #     member do
  #       get 'short'
  #       post 'toggle'
  #     end
  #
  #     collection do
  #       get 'sold'
  #     end
  #   end

  # Example resource route with sub-resources:
  #   resources :products do
  #     resources :comments, :sales
  #     resource :seller
  #   end

  # Example resource route with more complex sub-resources:
  #   resources :products do
  #     resources :comments
  #     resources :sales do
  #       get 'recent', on: :collection
  #     end
  #   end

  # Example resource route with concerns:
  #   concern :toggleable do
  #     post 'toggle'
  #   end
  #   resources :posts, concerns: :toggleable
  #   resources :photos, concerns: :toggleable

  # Example resource route within a namespace:
  #   namespace :admin do
  #     # Directs /admin/products/* to Admin::ProductsController
  #     # (app/controllers/admin/products_controller.rb)
  #     resources :products
  #   end
end
