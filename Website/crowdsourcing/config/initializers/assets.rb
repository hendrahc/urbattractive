# Be sure to restart your server when you modify this file.

# Version of your assets, change this if you want to expire all your assets.
Rails.application.config.assets.version = '1.0'

# Precompile additional assets.
# application.js, application.css, and all non-JS/CSS in app/assets folder are already added.
# Rails.application.config.assets.precompile += %w( search.js )

Rails.application.config.assets.precompile += %w( glyphicons-halflings.png )
Rails.application.config.assets.precompile += %w( custom.css )
Rails.application.config.assets.precompile += %w( bootstrap/bootstrap.css )
Rails.application.config.assets.precompile += %w( jquery-ui.min.css )
Rails.application.config.assets.precompile += %w( bootstrap/bootstrap.js )
Rails.application.config.assets.precompile += %w( external/jquery/jquery.js )
Rails.application.config.assets.precompile += %w( jquery-ui.min.js )
Rails.application.config.assets.precompile += %w( scales.js )
Rails.application.config.assets.precompile += %w( buttons.js )

Rails.application.config.assets.precompile += %w( external/jquery/jquery.mobile-1.1.0.min.js )
Rails.application.config.assets.precompile += %w( external/jquery/jquery.ui.affectbutton.js )
Rails.application.config.assets.precompile += %w( external/jquery/jquery.ui.affectbutton.mobile.js)

Rails.application.config.assets.precompile += %w( external/logger.js)
Rails.application.config.assets.precompile += %w( external/firebase.js)

Rails.application.config.assets.precompile += %w( jquery.mobile-1.0.1.min.css )